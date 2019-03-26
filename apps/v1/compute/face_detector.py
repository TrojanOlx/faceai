import sys
import dlib

detector=dlib.get_frontal_face_detector()
win=dlib.image_window()

img=dlib.load_rgb_image("./headPose1.jpg")
detected=detector(img,1)



if len(detected) > 0:
        for i, d in enumerate(detected):
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            print("{}{}{}{}".format(x1,y1,x2,y2))
        win.clear_overlay()
        win.set_image(img)
        win.add_overlay(detected)
        dlib.hit_enter_to_continue()
