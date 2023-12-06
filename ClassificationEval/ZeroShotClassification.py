import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import open_clip

# Opens the model based on which model to use and what the pretrained name is
# Returns the tokenizer, model, and preprocess (used for preprocessing input images)
def getModel(modelName, pretrainedName):
  model, _, preprocess = open_clip.create_model_and_transforms(modelName, pretrained=pretrainedName)
  tokenizer = open_clip.get_tokenizer(modelName)
  return model, tokenizer, preprocess

# Takes in the list of prompts, classes and then embeds for every prompt, class pair into the CLIP text encoder
# outputs all of the text emebeddings
def getTextEmbeddings(prompts, classes, model, tokenizer):
  textEmbeddings = None
  model_outputs = []
  with torch.no_grad():
    # iterate through all pairs of prompts/classes
    for promptType in prompts:
      promptList = []
      for classType in classes:
        promptEngineered = promptType + classType
        promptList.append(promptEngineered)
      # tokenize the prompts
      tokenizePrompt = tokenizer(promptList).to(device)
      # get the embeddings
      textEmbeddings = model.encode_text(tokenizePrompt)
      textEmbeddings = textEmbeddings / torch.linalg.vector_norm(textEmbeddings, dim = 1, keepdim = True)
      model_outputs.append(textEmbeddings)
  return model_outputs

# perform zero shot classifaction for input images
# returns similarity scores of prompt per class
def zeroShotClassification(images, model, allTextEmbeddings, device):
  imageEmbedding = None
  with torch.no_grad():
      # embed the images
      imageEmbedding = model.encode_image(images)
      imageEmbedding = imageEmbedding / torch.linalg.vector_norm(imageEmbedding, dim = 1, keepdim = True)
  # get similarity score per text embedding
  model_outputs = []
  with torch.no_grad():
    for textEmbeddings in allTextEmbeddings:
      model_outputs.append(imageEmbedding.mm(textEmbeddings.T))

  return torch.stack(model_outputs)

# returns the correct output class for each image
def getLabel(output):
  # keep track of score for each class label per prompts
  imageCount = torch.zeros((output.shape[1], output.shape[2]))
  for prompt in output:
    rowIndex = 0
    for imageRow in prompt:
      # take the softmax of each image row to get highest scoring label for that prompt
      softmaxOutput = torch.softmax(imageRow, dim = -1)
      index = torch.argmax(softmaxOutput)
      # increase counter for that label for the image
      imageCount[rowIndex][index] += 1
      rowIndex += 1
  ans = []
  # compute per image highest label
  for image in imageCount:
    index = torch.argmax(image)
    ans.append(index)
  return torch.tensor(ans).reshape(-1)

# Evaluate accuracy of zero shot classification for CIFAR10
def evaluateCIFAR10(prompts, classes, model, tokenizer, preprocess, device):
  # Get data
  dataset1 = datasets.CIFAR10('./data/', train=True, download=True,
                      transform=preprocess)
  dataset2 = datasets.CIFAR10('./data/', train=False,
                      transform=preprocess)
  train_loader = DataLoader(dataset1, batch_size = 20,
                              shuffle=True, num_workers=4)
  test_loader = DataLoader(dataset2, batch_size = 20,
                              shuffle=False, num_workers=4)
  # Get embeddings
  embeddings = getTextEmbeddings(prompts, classes, model, tokenizer)

  # Computer accuracy
  correct = 0
  # Enumerate train/test data of CIFAR10 and compute zero shot classification for it
  for batch_idx, batch_sample in enumerate(train_loader):
    data, target = batch_sample
    data = data.to(device)
    target = target.to(device)
    ans = zeroShotClassification(data, model, embeddings, device)
    output_labels = getLabel(ans).to(device)
    correct += torch.sum(torch.eq(output_labels, target))

  size = batch_idx + 1

  for batch_idx, batch_sample in enumerate(test_loader):
    data, target = batch_sample
    data = data.to(device)
    target = target.to(device)
    ans = zeroShotClassification(data, model, embeddings, device)
    output_labels = getLabel(ans).to(device)
    correct += torch.sum(torch.eq(output_labels, target))

  size += batch_idx + 1

  accuracy = correct / (size * 20)

  print(accuracy)

# Evaluate accuracy of zero shot classification for MNIST
def evaluateMNIST(prompts, classes, model, tokenizer, preprocess, device):
  dataset1 = datasets.MNIST('./data/', train=True, download=True,
                      transform=preprocess)
  dataset2 = datasets.MNIST('./data/', train=False,
                      transform=preprocess)
  train_loader = DataLoader(dataset1, batch_size = 20,
                              shuffle=True, num_workers=4)
  test_loader = DataLoader(dataset2, batch_size = 20,
                              shuffle=False, num_workers=4)

  embeddings = getTextEmbeddings(prompts, classes, model, tokenizer)

  correct = 0
  for batch_idx, batch_sample in enumerate(train_loader):
    data, target = batch_sample
    data = data.to(device)
    target = target.to(device)
    ans = zeroShotClassification(data, model, embeddings, device)
    output_labels = getLabel(ans).to(device)
    correct += torch.sum(torch.eq(output_labels, target))

  size = batch_idx + 1

  for batch_idx, batch_sample in enumerate(test_loader):
    data, target = batch_sample
    data = data.to(device)
    target = target.to(device)
    ans = zeroShotClassification(data, model, embeddings, device)
    output_labels = getLabel(ans).to(device)
    correct += torch.sum(torch.eq(output_labels, target))

  size += batch_idx + 1

  accuracy = correct / (size * 20)

  print(accuracy)

# Get model
model, tokenizer, preprocess = getModel('ViT-B-32', 'laion2b_s34b_b79k')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device).eval()

# Evaluate MNIST
prompts = ["is this the number ", "a image of the number ", "is this image blurry ",
           "is this number greater ", "is this number smaller ", "a handwritten ", "a computer generated "]
classes = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
evaluateMNIST(prompts, classes, model, tokenizer, preprocess, device)

# Evaluate CIFAR10
prompts = ["is this a number ", "is this bright ", "is this small ", "the object is a ", "how big are we talking for this "]
classes = ["airplanes", "cars", "birds", "cats", "deer", "dogs", "frogs", "horses", "ships", "trucks"]
evaluateCIFAR10(prompts, classes, model, tokenizer, preprocess, device)
