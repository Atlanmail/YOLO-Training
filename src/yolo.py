# %%
from transformers import AutoFeatureExtractor
from datasets import load_dataset, Image
import PIL
from transformers import YolosForObjectDetection
import torch
import matplotlib.pyplot as plt
import cv2

#%%
model = YolosForObjectDetection.from_pretrained("hustvl/yolos-small")
feature_extractor = AutoFeatureExtractor.from_pretrained("hustvl/yolos-small")
THRESHOLD = 0.9 # threshold for detecting an object

#%%
def detectObjects(image : PIL):


    pixel_values = feature_extractor(image, return_tensors="pt").pixel_values
    pixel_values.shape

    with torch.no_grad():
        outputs = model(pixel_values, output_attentions=True)
    
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > THRESHOLD

    # rescale bounding boxes
    
    target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
    postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
    bboxes_scaled = postprocessed_outputs[0]['boxes']

    return plot_results(image, probas[keep], bboxes_scaled[keep])
# %%
# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{model.config.id2label[cl.item()]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    
    plt.savefig("temp.jpg")
    img = cv2.imread("temp.jpg")
    return PIL.Image.fromarray(img)

# %%



