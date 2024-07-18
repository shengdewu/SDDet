import os
import cv2
import numpy as np
import tqdm


colors_rgb = [
    (0, 0, 0),
    (0, 255, 255),
    (255, 140, 0),
    (173, 255, 47),
    (128, 0, 128),
    (218, 165, 32),
    (205, 92, 92),
    (139, 0, 139),
    (255, 0, 0),
    (0, 0, 128),
    (255, 0, 255),
]


def find_non_zero(ctx):
    pos = -1
    for i, c in enumerate(ctx):
        pos = i
        if c != '0':
            break
    return pos


def create_celeba_mask():
    """
    处理CelebAMask-HQ数据集， 转换成满足训练要求的数据
    :return: 
    """
    in_root = '/mnt/sda2/脸部区域分割/celeba/CelebAMask-HQ'
    out_root = '/mnt/sda2/脸部区域分割/celeba/CelebAMask-Face'

    atts = ['skin', 'r_brow', 'l_brow', 'r_eye', 'l_eye', 'nose', 'u_lip', 'mouth', 'l_lip']

    mask_path = f'{in_root}/CelebAMask-HQ-mask-anno'
    img_path = f'{in_root}/CelebA-HQ-img'
    mask_names = dict()
    for part in os.listdir(mask_path):
        for name in os.listdir(f'{mask_path}/{part}'):
            if name.startswith('.'):
                continue

            i_name = name.split('_')[0]
            pos = find_non_zero(i_name)
            if pos != -1:
                i_name = i_name[pos:]

            if not os.path.exists(f'{img_path}/{i_name}.jpg'):
                print(f'not found {img_path}/{i_name}.jpg')
                continue

            m_name = '_'.join(name.split('_')[1:]).split('.')[0]
            if mask_names.get(i_name, None) is None:
                mask_names[i_name] = list()
            mask_names[i_name].append((m_name, f'{part}/{name}'))

    out_img_path = f'{out_root}/image'
    out_label_path = f'{out_root}/label'
    out_test_path = f'{out_root}/test'
    os.makedirs(out_img_path, exist_ok=True)
    os.makedirs(out_label_path, exist_ok=True)
    os.makedirs(out_test_path, exist_ok=True)

    for i_name, part in tqdm.tqdm(mask_names.items()):

        img = cv2.imread(f'{img_path}/{i_name}.jpg', cv2.IMREAD_COLOR)
        cv2.imwrite(f'{out_img_path}/{i_name}.jpg', img)

        h, w, c = img.shape
        label = np.zeros(shape=(h, w), dtype=img.dtype)

        alpha = np.zeros_like(img)
        dis_mask = np.zeros_like(img)

        assert len(part) == len(set(part))

        att_part = dict(part)

        skin = None
        max_idx = -1
        for i, att in enumerate(atts, 1):
            if att not in att_part:
                print(f'not found {i_name} - {att}')
                continue

            max_idx = i

            mask = cv2.imread(f'{mask_path}/{att_part[att]}', cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            thread = 125
            _, mask = cv2.threshold(mask, thread, i, cv2.THRESH_BINARY)
            if att == 'skin':
                skin = mask.copy()
                continue

            if skin is not None:
                skin[mask > 0] = 0

            label += mask

            if np.max(label) != i:
                label[label > i] = i

            alpha[..., 0] = colors_rgb[i][2]
            alpha[..., 1] = colors_rgb[i][1]
            alpha[..., 2] = colors_rgb[i][0]
            dis_mask[..., 0] = label
            dis_mask[..., 1] = label
            dis_mask[..., 2] = label

            tmp = cv2.addWeighted(img, 0.7, alpha, 0.3, 0)
            img = np.where(dis_mask == i, tmp, img)

        if skin is not None:
            label += skin
            assert np.max(label) <= max_idx

            alpha[..., 0] = colors_rgb[1][2]
            alpha[..., 1] = colors_rgb[1][1]
            alpha[..., 2] = colors_rgb[1][0]
            dis_mask[..., 0] = label
            dis_mask[..., 1] = label
            dis_mask[..., 2] = label

            tmp = cv2.addWeighted(img, 0.7, alpha, 0.3, 0)
            img = np.where(dis_mask == 1, tmp, img)

        assert np.max(label) < 10
        cv2.imwrite(f'{out_label_path}/{i_name}.png', label)
        cv2.imwrite(f'{out_test_path}/{i_name}.jpg', img)

    out_occlusion_path = f'{out_root}/occlusion'
    for i_name, part in tqdm.tqdm(mask_names.items()):
        att_part = dict(part)
        for att in atts:
            if att in att_part:
                continue
            out_path = f'{out_occlusion_path}/{att}'
            os.makedirs(out_path, exist_ok=True)
            cmd = f'cp {out_test_path}/{i_name}.jpg {out_path}/{i_name}.jpg'
            os.system(cmd)

    return


def create_helen_mask():
    """
    处理helen数据集， 转换成满足训练要求的数据, 通过背景过滤多人
    :return:
    """
    root_path = '/mnt/sda2/脸部区域分割/helen/helen'
    label_path = f'{root_path}/labels'
    image_path = f'{root_path}/image'

    out_root = '/mnt/sda2/脸部区域分割/helen/helen-face'
    out_img_path = f'{out_root}/image'
    out_label_path = f'{out_root}/label'
    out_test_path = f'{out_root}/test'
    os.makedirs(out_img_path, exist_ok=True)
    os.makedirs(out_label_path, exist_ok=True)
    os.makedirs(out_test_path, exist_ok=True)

    for name in os.listdir(image_path):
        img = cv2.imread(f'{image_path}/{name}', cv2.IMREAD_COLOR)
        name_pre = name.split('.')[0]
        back = cv2.imread(f'{label_path}/{name_pre}/{name_pre}_lbl00.png', cv2.IMREAD_GRAYSCALE)
        hair = cv2.imread(f'{label_path}/{name_pre}/{name_pre}_lbl10.png', cv2.IMREAD_GRAYSCALE)
        mask = np.bitwise_and(255 - hair, 255 - back)
        mask[mask > 0] = 255

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour
        x, y, w, h = cv2.boundingRect(max_contour)
        mh, mw, _ = img.shape
        extend = 1.2
        nw = int(w * extend + 0.5)
        nh = int(h * extend + 0.5)
        dw = int((nw - w) * 0.5)
        dh = int((nh - h) * 0.5)
        x1 = max(0, x - dw)
        y1 = max(0, y - dh)
        x2 = min(x + nw, mw)
        y2 = min(y + nh, mh)
        cut_img = img[y1:y2, x1:x2, :]
        cv2.imwrite(f'{out_img_path}/{name}', cut_img)
        h, w, c = cut_img.shape
        label = np.zeros((h, w), dtype=mask.dtype)
        mask_name = f'{label_path}/{name_pre}/{name_pre}_lbl'
        alpha = np.zeros_like(cut_img)
        mask = np.zeros_like(cut_img)
        for i in range(1, 10):
            back = cv2.imread("%s%02d.png" % (mask_name, i), cv2.IMREAD_GRAYSCALE)
            if back is None:
                print(f'the {mask_name} in {i} not found')
                continue

            # thread = np.mean(back) + np.max(back) * 0.1
            thread = 125
            _, back = cv2.threshold(back, thread, i, cv2.THRESH_BINARY)
            label += back[y1:y2, x1:x2]
            if np.max(label) != i:
                label[label > i] = i
            alpha[..., 0] = colors_rgb[i][2]
            alpha[..., 1] = colors_rgb[i][1]
            alpha[..., 2] = colors_rgb[i][0]
            mask[..., 0] = label
            mask[..., 1] = label
            mask[..., 2] = label
            tmp = cv2.addWeighted(cut_img, 0.3, alpha, 0.7, 0)
            cut_img = np.where(mask == i, tmp, cut_img)
            # cv2.imwrite(f'{out_test_path}/{i}-{name}', tmp)

        assert np.max(label) < 10
        cv2.imwrite(f'{out_label_path}/{name_pre}.png', label)
        cv2.imwrite(f'{out_test_path}/{name}', cut_img)
    return


def split(in_root, img_part, label_part, flag):
    import random
    names = os.listdir(f'{in_root}/{img_part}')
    train_names = random.sample(names, k=int(0.9 * len(names)))
    with open(f'{in_root}/{flag}-train.txt', mode='w') as tf:
        with open(f'{in_root}/{flag}-valid.txt', mode='w') as vf:
            for name in names:
                assert name.endswith('jpg')
                lname = name.replace('jpg', 'png')
                if name in train_names:
                    tf.write(f'{img_part}/{name},{label_part}/{lname}\n')
                else:
                    vf.write(f'{img_part}/{name},{label_part}/{lname}\n')
    return

if __name__ == '__main__':
    # create_celeba_mask()
    # create_helen_mask()
    split('/mnt/sda2/脸部区域分割/helen/helen-face', 'occ-image', 'occ-label', 'occ')
    split('/mnt/sda2/脸部区域分割/celeba/CelebAMask-Face', 'occ-image', 'occ-label', 'occ')