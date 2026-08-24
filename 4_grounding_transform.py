import cv2

def ground_data(image, question, bbox):
    x, y, w, h = bbox

    grounded_img = image.copy()
    cv2.rectangle(grounded_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    grounded_q = question + " Focus only on the object inside the red bounding box."

    return grounded_img, grounded_q