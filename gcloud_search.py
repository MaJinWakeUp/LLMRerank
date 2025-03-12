import os
from google.cloud import vision
import json

def detect_web(path):
    """Detects web annotations given an image."""
    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.web_detection(image=image)
    annotations = response.web_detection

    """
    if annotations.best_guess_labels:
        for label in annotations.best_guess_labels:
            print(f"\nBest guess label: {label.label}")

    if annotations.pages_with_matching_images:
        print(
            "\n{} Pages with matching images found:".format(
                len(annotations.pages_with_matching_images)
            )
        )

        for page in annotations.pages_with_matching_images:
            print(f"\n\tPage url   : {page.url}")

            if page.full_matching_images:
                print(
                    "\t{} Full Matches found: ".format(len(page.full_matching_images))
                )

                for image in page.full_matching_images:
                    print(f"\t\tImage url  : {image.url}")

            if page.partial_matching_images:
                print(
                    "\t{} Partial Matches found: ".format(
                        len(page.partial_matching_images)
                    )
                )

                for image in page.partial_matching_images:
                    print(f"\t\tImage url  : {image.url}")

    if annotations.web_entities:
        print("\n{} Web entities found: ".format(len(annotations.web_entities)))

        for entity in annotations.web_entities:
            print(f"\n\tScore      : {entity.score}")
            print(f"\tDescription: {entity.description}")

    if annotations.visually_similar_images:
        print(
            "\n{} visually similar images found:\n".format(
                len(annotations.visually_similar_images)
            )
        )

        for image in annotations.visually_similar_images:
            print(f"\tImage url    : {image.url}")

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )
    """

    results = {}

    if annotations.best_guess_labels:
        results["best_guess_labels"] = [label.label for label in annotations.best_guess_labels]

    if annotations.pages_with_matching_images:
        results["pages_with_matching_images"] = []
        for page in annotations.pages_with_matching_images:
            page_info = {
                "url": page.url,
                "full_matching_images": [image.url for image in page.full_matching_images],
                "partial_matching_images": [image.url for image in page.partial_matching_images],
            }
            results["pages_with_matching_images"].append(page_info)

    if annotations.web_entities:
        results["web_entities"] = [
            {"score": entity.score, "description": entity.description}
            for entity in annotations.web_entities
        ]

    if annotations.visually_similar_images:
        results["visually_similar_images"] = [image.url for image in annotations.visually_similar_images]

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

    return results


if __name__ == "__main__":
    save_dir = "./gcloud_output"
    images_dir = "/scratch/jin7/datasets/AMMeBa/images_part1/"
    demo_image_id = "68a6062ab9404ffdc6fc84212b83008936ab06d07cf3503d95262db4"
    demo_image_path = os.path.join(images_dir, demo_image_id + ".jpg")
    res = detect_web(demo_image_path)
    output_file = os.path.join(save_dir, f"{demo_image_id}.json")
    with open(output_file, "w") as f:
        json.dump(res, f, indent=4)