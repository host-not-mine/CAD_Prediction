from typing import Union

import torch.utils.data

from .Dataloader import AngiogramDataset, ImageLoader
from .Unet import CoronarySegmentationModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
imageLoader = ImageLoader(device)
BATCH = 1


def calculate_iou(pred:torch.Tensor, target:torch.Tensor):
    ious = []
    for i in range(pred.shape[0]):  # Iterate through images in the batch
        # for j in range(pred.shape[1]):  # Iterate through predicted masks
            j = 0
            pred_mask = pred[i, j, :, :]
            target_mask = target[i, j, :, :]

            target_mask:torch.Tensor = 0 # type: ignore
            for m in target[i] :
                target_mask += m

            intersection = (pred_mask * target_mask).sum(dim=(0, 1))
            union = pred_mask.sum(dim=(0, 1)) + target_mask.sum(dim=(0, 1)) - intersection
            iou = (intersection + 1e-6) / (union + 1e-6)  # Small value for stability
            ious.append(iou)
    return sum(ious) / len(ious)

def calculate_dice_score(pred:torch.Tensor, target:torch.Tensor):
    scores = []
    for i in range(pred.shape[0]):
        # for j in range(pred.shape[1]):
            j = 0
            pred_mask = pred[i, j, :, :]
            target_mask = target[i, j, :, :]

            target_mask:torch.Tensor = 0 # type: ignore
            for m in target[i] :
                target_mask += m

            intersection = (pred_mask * target_mask).sum(dim=(0, 1))
            union = pred_mask.sum(dim=(0, 1)) + target_mask.sum(dim=(0, 1)) 
            dice = (2 * intersection + 1e-6) / (union + 1e-6)  # Small value for stability
            scores.append(dice)
    return sum(scores) / len(scores)

def calculate_specificity_sensitivity(pred:torch.Tensor, target:torch.Tensor):
    sensitivity_scores = []
    specificity_score = []
    for i in range(pred.shape[0]):
        # for j in range(pred.shape[1]):
            j = 0
            pred_mask = pred[i, j, :, :]
            target_mask = target[i, j, :, :]

            target_mask:torch.Tensor = 0 # type: ignore
            for m in target[i] :
                target_mask += m

            true_positive = (pred_mask * target_mask).sum(dim=(0, 1))  # Correctly predicted positive pixels
            true_negative = ((1 - pred_mask) * (1 - target_mask)).sum(dim=(0, 1))  # Correctly predicted negative pixels
            false_positive = (pred_mask * (1 - target_mask)).sum(dim=(0, 1))  # Incorrectly predicted positive pixels
            false_negative = ((1 - pred_mask) * target_mask).sum(dim=(0, 1))  # Incorrectly predicted negative pixels

            sensitivity = (true_positive + 1e-6) / (true_positive + false_negative + 1e-6)
            specificity = (true_negative + 1e-6) / (true_negative + false_positive + 1e-6)
            sensitivity_scores.append(sensitivity)
            specificity_score.append(specificity)
    sensitivity_scores = sum(sensitivity_scores)/len(sensitivity_scores)
    specificity_score = sum(specificity_score)/len(specificity_score)
    return sensitivity_scores, specificity_score

def calculate_metrics(preds:torch.Tensor, masks:torch.Tensor) -> dict[str, float]:
    metrics_dict = {}
    metrics_dict["IoU"] = calculate_iou(preds, masks)
    metrics_dict["Dice_Score"] = calculate_dice_score(preds, masks)

    sensitivity, specificity = calculate_specificity_sensitivity(preds, masks)
    metrics_dict["Sensitivity"] = sensitivity
    metrics_dict["Specificity"] = specificity

    return metrics_dict


def load_model(model_path:str, max_masks=1):
    try:
        model_test = CoronarySegmentationModel.load_from_checkpoint(checkpoint_path=model_path, max_masks=max_masks)
    # except RuntimeError:
    #     MAX_MASKS = 2
    #     model_test = CoronarySegmentationModel.load_from_checkpoint(checkpoint_path=model_path)
    except AttributeError:
        raise FileNotFoundError("Version not found")

    model_test.eval()
    model_test.to(device)
    return model_test


def get_dataloader(demo_image_dir, max_masks=1):
    global dataset_test
 
    dataset_test = AngiogramDataset(demo_image_dir, load_to_device=device, max_masks=max_masks)
    test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH, shuffle=False)  # Adjust batch size

    return test_dataloader

# shape (batch, channels, h, w)
def stack_masks(preds:torch.Tensor, shape:Union[torch.Size, tuple]=(BATCH, 1, 512, 512)):
    cated = torch.zeros(shape, dtype=torch.float32, device=device)
    for i,val in enumerate(preds):
        pred_mask = 0
        for j in val:
            pred_mask += j
        cated[i] += pred_mask

    return cated

def evaluate_model(model_path:str, demo_image_dir:str, sigmoid=False, max_masks=1):
    model = load_model(model_path, max_masks=max_masks)
    dataloader = get_dataloader(demo_image_dir, max_masks=max_masks)

    all_metrics = []
    for image, mask in dataloader:
        with torch.no_grad():
            preds = model(image.to(device))

            if sigmoid:  # Apply Sigmoid if needed
                preds = torch.sigmoid(preds)

            preds = stack_masks(preds)
            # mask = stack_masks(mask) # try this on CAD_site

            metrics = calculate_metrics(preds, mask)
            all_metrics.append(metrics) 

    # Calculate and report average metrics
    average_iou = sum(m["IoU"] for m in all_metrics) / len(all_metrics)
    average_dice = sum(m["Dice_Score"] for m in all_metrics) / len(all_metrics)
    average_sensitivity = sum(m["Sensitivity"] for m in all_metrics) / len(all_metrics)
    average_specificity = sum(m["Specificity"] for m in all_metrics) / len(all_metrics)

    print(f"Average IoU: {average_iou}")
    print(f"Average Dice Score: {average_dice}")
    print(f"Average Sensitivity: {average_sensitivity}")
    print(f"Average Specificity: {average_specificity}")

    return {
        "IoU_avg" : float(average_iou.cpu()),  # type: ignore
        "Dice avg": float(average_dice.cpu()), # type: ignore
        "Sensitivity_avg": float(average_sensitivity.cpu()),  # type: ignore
        "Specificity_avg": float(average_specificity.cpu()),  # type: ignore
        "learning_rate": model.learning_rate
    }

class ModelStore:
    _model:dict[int, CoronarySegmentationModel] = {}
    @classmethod
    def load_model(cls, name, path, mask_limit=1) -> CoronarySegmentationModel:
        model = load_model(path, max_masks=mask_limit)
        cls._model[name] = model
        return model
    @classmethod
    def get_model(cls, name):
        try:
            model = cls._model[name]
        except KeyError:
            raise NameError(f"No model named: {name} loaded.")
        return model
    @classmethod
    def list_model(cls):
        return cls._model.keys()

def predict(model_path, image_path, max_masks=1):
    model = load_model(model_path, max_masks=max_masks)
    image = imageLoader.load(image_path)
    with torch.no_grad():
        image = image.unsqueeze(0)
        pred = model(image)
    print("Pred shape: ", pred.shape)
    pred = imageLoader.convert_to_pil(pred[0])
    return pred

def test_image(model:CoronarySegmentationModel, image_path:str):
    image = imageLoader.load(image_path)

    with torch.no_grad():
        image = image.unsqueeze(0)
        pred = model(image)
    pred = imageLoader.convert_to_pil(pred[0])
    return pred

