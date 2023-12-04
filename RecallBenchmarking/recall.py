# Recall Script with modifications to custom annotation file and using open clip and finetuned models
# Note: We base our Recall Score test script on this post https://github.com/openai/CLIP/issues/115

import torch
from torchvision.datasets import CocoCaptions
from torch.utils.data import DataLoader
import open_clip
from tqdm import tqdm

CAPTIONS_PER_IMAGE = 5 # Note: This is because some images in the COCO captions dataset have more than 5 captions, so we cap it at 5
RUN_FINETUNED_CLIP = True
COCO_ROOT_PATH = r"D:\Programming\datasets\coco\val2017"
COCO_ANNOTATIONS_PATH = r"D:\Programming\CAP 5415\ClipSemanticSearch\RecallBenchmarking\annotation_files\captions_val2017_1000_subset.json"
USE_EUCLIDEAN_DISTANCE = False

print(f"{USE_EUCLIDEAN_DISTANCE=}")

# Load our model for testing
if RUN_FINETUNED_CLIP:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    model_name = 'ViT-B-32'
    finetuned_path = 'D:\Programming\CAP 5415\ClipSemanticSearch\pretrained_model\clip_coco_train2017_epoch4.pt'

    print(f"Finetuned Model: {model_name=}, {finetuned_path=}")

    # We load in the model ourselves so we can correctly obtain the model weights
    checkpoint = torch.load(finetuned_path)
    state_dict = checkpoint['model_state_dict']

    # Create the randomly initialized CLIP model and then load in the state dict
    model, _, transform = open_clip.create_model_and_transforms(model_name)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    tokenizer = open_clip.get_tokenizer(model_name)
# Run the Base CLIP model
else:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    model_name = 'RN50'
    pretrained_name = 'openai'
    print(f"{model_name=}, {pretrained_name=}")

    model, _, transform = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_name)
    model.to(device).eval()

    tokenizer = open_clip.get_tokenizer(model_name)

# K values for recall test
k_vals =[1, 5, 10, 50]

dataset = CocoCaptions(
    root=COCO_ROOT_PATH,
    annFile=COCO_ANNOTATIONS_PATH,
    transform=transform,
    # Note: almost all images have 5 captions, some have 6 or 7. We ignore the extras
    target_transform=lambda texts: tokenizer(texts[:CAPTIONS_PER_IMAGE])
)

def encode_dataset(model, dataset, batch_size=16):
    with torch.no_grad():
        # Mappings for the encodings from text-images and vice versa
        text_to_image_map = []
        image_to_text_map = []

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        image_encodings, text_encodings = [], []

        text_index, image_index = 0, 0

        # Update text_to_image_map and image_to_text_map for this batch
        print("Generating Text-Image and Image-Text Mappings")
        for _, text in tqdm(dataloader):
            batch_size = text.shape[0]
            for _ in range(batch_size):
                # Each of the next CAPTIONS_PER_IMAGE text captions correspond to the same image
                text_to_image_map += [image_index for _ in range(CAPTIONS_PER_IMAGE)]
                image_index += 1

                # the next image corresponds to text captions [text_index ... text_index + captions_per_image - 1]
                text_indices = [index for index in range(text_index, text_index + CAPTIONS_PER_IMAGE)]
                image_to_text_map.append(text_indices)
                text_index += CAPTIONS_PER_IMAGE

        print(f"T-I Mapping Length: {len(text_to_image_map)}")        
        print(f"I-T Mapping Length: {len(image_to_text_map)}")

        print('Obtaining Encodings')
        for images, text in tqdm(dataloader):
            images = images.to(device)
            text = text.to(device) # Note: Text shape: (B x 5 x 77)

            # B x 5 x 77 -> (B*5) x 77
            text = torch.flatten(text, start_dim=0, end_dim=1)
            
            # Get the encodings
            image_encodings.append(model.encode_image(images))
            text_encodings.append(model.encode_text(text))

        image_encodings = torch.cat(image_encodings)
        text_encodings = torch.cat(text_encodings)
        text_to_image_map = torch.LongTensor(text_to_image_map).to(device)
        image_to_text_map = torch.LongTensor(image_to_text_map).to(device)

        print(f"Image_Encodings Shape: {image_encodings.shape}")
        print(f"Text_Encodings Shape: {text_encodings.shape}")
        print(f"T-I Map Shape: {text_to_image_map.shape}")
        print(f"I-T Map Shape: {image_to_text_map.shape}")

        return image_encodings, text_encodings, text_to_image_map, image_to_text_map

def recall_at_k(model, dataset, k_vals, batch_size=16):
    print("Encoding Dataset")
    image_encodings, text_encodings, text_to_image_map, image_to_text_map = encode_dataset(model, dataset, batch_size=batch_size)
 
    num_text = text_encodings.shape[0]
    num_im = image_encodings.shape[0]
    print(f"{num_text=}, {num_im=}")

    # text-to-image recall
    print("Text-to-image recall")

    if USE_EUCLIDEAN_DISTANCE:
        print("Using Euclidean Distance (Note: We are not using normalized vectors)")
        '''
        Note: If we normalized the vectors then the Euclidean similarity rankings would be equivalent to the 
            cosine similarity: https://stats.stackexchange.com/questions/146221/is-cosine-similarity-identical-to-l2-normalized-euclidean-distance
        '''
        dist_matrix = torch.cdist(text_encodings.unsqueeze(0), image_encodings.unsqueeze(0), p=2)
        dist_matrix = 1 / (1 + dist_matrix) # Find the inverse of the distance to find similarity
        dist_matrix = dist_matrix.squeeze(0)
    else:
        print("Using Cosine Similarity")
        '''
        Obtain the Cosine similarity by multiplying the normalized vectors

        Note: Cosine(x, y) = (x * y) / (||x|| * ||y||), but if the vectors are already normalized, then 
            ||x|| * ||y|| = 1, and thus the formula becomes: cosine(x, y) = (x * y) for normalized vectors   
        '''

        # Normalize the vectors
        image_encodings = image_encodings / image_encodings.norm(dim=-1, keepdim=True)
        text_encodings = text_encodings / text_encodings.norm(dim=-1, keepdim=True)

        dist_matrix = text_encodings @ image_encodings.T  # dist_matrix[i] gives logits for ith text

    # Note: this matrix is pretty big (5000 x 25000 with dtype float16 = 250MB)
    #  torch.argsort runs out of memory for me (6GB VRAM) so I move to CPU for sorting
    dist_matrix = dist_matrix.cpu()

    print(f'dist_matrix shape: {dist_matrix.shape}')

    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.to(device)

    text_to_image_recall = []

    print("Obtaining Scores for each K value")
    for k in tqdm(k_vals):
        # Extract top k indices only
        topk = inds[:, :k]

        # Correct iff one of the top_k values equals the correct image (as given by text_to_image_map)
        correct = torch.eq(topk, text_to_image_map.unsqueeze(-1)).any(dim=1)

        num_correct = correct.sum().item()
        text_to_image_recall.append(num_correct / num_text)

    # image-to-text recall
    print("Image-to-text recall")
    dist_matrix = dist_matrix.T  # dist_matrix[i] gives logits for the ith image

    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.to(device)

    image_to_text_recall = []

    print("Obtaining Scores for each K value")
    for k in tqdm(k_vals):
        # Extract top k indices only
        topk = inds[:, :k]

        correct = torch.zeros((num_im,), dtype=torch.bool).cuda()

        #  For each image, check whether one of the 5 relevant captions was retrieved
        # Check if image matches its ith caption (for i=0..4)
        for i in range(CAPTIONS_PER_IMAGE):
            contains_index = torch.eq(topk, image_to_text_map[:, i].unsqueeze(-1)).any(dim=1)
            correct = torch.logical_or(correct, contains_index)

        num_correct = correct.sum().item()
        image_to_text_recall.append(num_correct / num_im)#

    print("Done")
    return text_to_image_recall, image_to_text_recall

t2i, i2t = recall_at_k(model, dataset, k_vals=k_vals, batch_size=16)

print("Text-to-image Recall@K")
for k, score in zip(k_vals, t2i):
    print(f" R@{k}: {100*score:.2f}%")

print("Image-to-text Recall@K")
for k, score in zip(k_vals, i2t):
    print(f" R@{k}: {100*score:.2f}%")