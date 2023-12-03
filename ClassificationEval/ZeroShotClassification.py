import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import open_clip

def getModel(modelName, pretrainedName):
  model, _, preprocess = open_clip.create_model_and_transforms(modelName, pretrained=pretrainedName)
  tokenizer = open_clip.get_tokenizer(modelName)
  return model, tokenizer, preprocess

def getTextEmbeddings(prompts, classes, model, tokenizer):
  textEmbeddings = None
  model_outputs = []
  with torch.no_grad():
    for promptType in prompts:
      promptList = []
      for classType in classes:
        promptEngineered = promptType + classType
        promptList.append(promptEngineered)
      tokenizePrompt = tokenizer(promptList).to(device)
      textEmbeddings = model.encode_text(tokenizePrompt)
      textEmbeddings = textEmbeddings / torch.linalg.vector_norm(textEmbeddings, dim = 1, keepdim = True)
      model_outputs.append(textEmbeddings)
  return model_outputs

def zeroShotClassification(images, model, allTextEmbeddings, device):
  imageEmbedding = None
  with torch.no_grad():
      imageEmbedding = model.encode_image(images)
      imageEmbedding = imageEmbedding / torch.linalg.vector_norm(imageEmbedding, dim = 1, keepdim = True)

  model_outputs = []
  with torch.no_grad():
    for textEmbeddings in allTextEmbeddings:
      model_outputs.append(imageEmbedding.mm(textEmbeddings.T))

  return torch.stack(model_outputs)

def getLabel(output):
  imageCount = torch.zeros((output.shape[1], output.shape[2]))
  for prompt in output:
    rowIndex = 0
    for imageRow in prompt:
      softmaxOutput = torch.softmax(imageRow, dim = -1)
      index = torch.argmax(softmaxOutput)
      imageCount[rowIndex][index] += 1
      rowIndex += 1
  ans = []
  for image in imageCount:
    index = torch.argmax(image)
    ans.append(index)
  return torch.tensor(ans).reshape(-1)

def evaluateCIFAR10(prompts, classes, model, tokenizer, preprocess, device):
  dataset1 = datasets.CIFAR10('./data/', train=True, download=True,
                      transform=preprocess)
  dataset2 = datasets.CIFAR10('./data/', train=False,
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

model, tokenizer, preprocess = getModel('ViT-B-32', 'laion2b_s34b_b79k')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device).eval()

prompts = ["is this the number ", "a image of the number ", "is this image blurry ",
           "is this number greater ", "is this number smaller ", "a handwritten ", "a computer generated "]
classes = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
evaluateMNIST(prompts, classes, model, tokenizer, preprocess, device)

prompts = ["is this a number ", "is this bright ", "is this small ", "the object is a ", "how big are we talking for this "]
classes = ["airplanes", "cars", "birds", "cats", "deer", "dogs", "frogs", "horses", "ships", "trucks"]
evaluateCIFAR10(prompts, classes, model, tokenizer, preprocess, device)
