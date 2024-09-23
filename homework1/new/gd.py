import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import VGG13_BN_Weights, vgg13_bn
from tqdm import tqdm
import torch.nn.functional as F

DEVICE = "cuda"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def save_img(image, path):
    # Push to CPU, convert from (1, 3, H, W) into (H, W, 3)
    image = image[0].permute(1, 2, 0)
    image = image.clamp(min=0, max=1)
    image = (image * 255).cpu().detach().numpy().astype(np.uint8)
    # opencv expects BGR (and not RGB) format
    cv.imwrite(path, image[:, :, ::-1])


def main():
    model = vgg13_bn(VGG13_BN_Weights.IMAGENET1K_V1).to(DEVICE)
    print(model)
    for label in [0, 12, 954]:
        image = torch.randn(1, 224, 224, 3).to(DEVICE)
        image = (image * 8 + 128) / 255  # background color = 128,128,128
        image = image.permute(0, 3, 1, 2)
        image.requires_grad_()
        image = gradient_descent(
            image,
            model,
            lambda tensor: tensor[0, label].mean(),
        )
        save_img(image, f"./img_{label}.jpg")
        out = model(image)
        print(f"ANSWER_FOR_LABEL_{label}: {out.softmax(1)[0, label].item()}")


# DO NOT CHANGE ANY OTHER FUNCTIONS ABOVE THIS LINE FOR THE FINAL SUBMISSION


def normalize_and_jitter(img, step=32):
    # You should use this as data augmentation and normalization,
    # convnets expect values to be mean 0 and std 1
    dx, dy = np.random.randint(-step, step - 1, 2)
    return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(
        img.roll(dx, -1).roll(dy, -2)
    )


def gaussianBlur(img, kernel_size=3, sigma=0.5):
    np_img = img.squeeze().permute(1, 2, 0).cpu().numpy()

    # applying blur from cv library -- guess params
    blurred = cv.GaussianBlur(np_img, (kernel_size, kernel_size), sigma)

    return torch.from_numpy(blurred).permute(2, 0, 1).unsqueeze(0).to(img.device)


def gradient_descent(input, model, loss, iterations=256):
    scales = 4
    min_size = 32 # turns out 4 scales reduces image size too much to be processed so change dim 28 to 32
    # extracting just the H and W values from the shape tuple, ignoring the batch size and number of channels
    original_size = input.shape[2:]

    for scale in range(scales):
        # resizing input here at every scale of image
        current_size =  [max(dim // (2**(scales-scale-1)), min_size) for dim in original_size]
        current_pixels = F.interpolate(
            input, size=current_size, mode="bilinear", align_corners=False
        )
        """
        Scale 0: 224 // (2**(4-0-1)) = 224 // 8 = 28 <<< changing to 32 (see above)
        Scale 1: 224 // (2**(4-1-1)) = 224 // 4 = 56
        Scale 2: 224 // (2**(4-2-1)) = 224 // 2 = 112
        Scale 3: 224 // (2**(4-3-1)) = 224 // 1 = 224
        (32, 32) -> (56, 56) -> (112, 112) -> (224, 224)
        """

        # resize the image in each scale, so want the resized image to be treated as a new variable
        # thus remove tree history from previous iterations (from earlier scales or gradient updates)
        current_pixels = current_pixels.detach().requires_grad_(True)

        optimizer = torch.optim.Adam([current_pixels], lr=0.05, weight_decay=1e-4)

        # with 4 scales, get 64 iterations
        for _ in tqdm(range(iterations // scales), desc=f"Scale {scale+1}/{scales}"):
            optimizer.zero_grad()

            # normalize and jitter the input
            normalized_input = normalize_and_jitter(current_pixels)

            output = model(normalized_input)

            # want to maximize, so negate the loss
            loss_value = -loss(output)

            loss_value.backward()

            # blur gradients
            if current_pixels.grad is not None:
                current_pixels.grad.data = gaussianBlur(
                    current_pixels.grad.data, kernel_size=3, sigma=0.5
                )

            optimizer.step()

            # clamp pixels between 0 and 1
            current_pixels.data.clamp_(0, 1)

            # blur the image as well!
            current_pixels.data = gaussianBlur(
                current_pixels.data, kernel_size=3, sigma=0.5
            )

        # update the original input with the result from this scale
        input = current_pixels.detach()

        # resize back to original size if not in the last scale
        if scale < scales - 1:
            input = F.interpolate(
                input, size=original_size, mode="bilinear", align_corners=False
            )

    return input


def forward_and_return_activation(model, input, module):
    """
    This function is for the extra credit. You may safely ignore it.
    Given a module in the middle of the model (like `model.features[20]`),
    it will return the intermediate activations.
    Try setting the model to `model.features[20]` and the loss to `tensor[0, ind].mean()`
    to see what intermediate activations activate on.
    """
    features = []

    def hook(model, input, output):
        features.append(output)

    handle = module.register_forward_hook(hook)
    model(input)
    handle.remove()

    return features[0]


if __name__ == "__main__":
    main()
