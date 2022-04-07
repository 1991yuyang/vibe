import numpy as np
from numpy import random as rd
import cv2


N = 25  # 每个像素的背景样本集合容量
R = 35  # 定义以当前像素值为中心的球体半径范围
Umin = 2  # 定义在球体范围内的背景模型元素数目阈值，如果大于阈值则像素点为背景，否则为前景
frame_h = 120
frame_w = 160
neighborhoods_count = 8  # 指定当前像素几邻域
phai = 16
foreground_pixel_keep_times = 20  # 前景像素点连续为foreground_pixel_keep_times次前景时将其变为背景像素点
pixel_update_prob = 1 / phai
filter_size = int(np.sqrt(neighborhoods_count + 1))  # 邻域搜索尺寸，例如8邻域的搜索尺寸为3 * 3
pad_size = int((filter_size - 1) / 2)  # 提取帧的第一个像素的邻域需要对边界进行padding，指定每个边的padding尺寸


def calc_background_samples_keep_prob(background_samples_keep_times, N):
    """
    计算每个样本不被更新的概率
    :param background_samples_keep_times: 记录背景样本集中每个样本连续没有得到更新的次数
    :param N: 每个像素的背景样本集合容量
    :return: background_samples_keep_prob，形状为(frame_h, frame_w, N)，元素代表当前背景样本被保留的概率
    """
    background_samples_keep_prob = np.power((N - 1) / N, background_samples_keep_times)
    return background_samples_keep_prob


def init_background_samples(gray_frame, N):
    """
    初始化每个像素的背景样本集
    :param gray_frame: 第一帧
    :param N: 每个像素的背景样本集合容量
    :return: background_samples，形状为(frame_h, frame_w, N)
            background_times，形状为(frame_h, frame_w)，记录当前帧的每个像素连续多少次判定为背景
            foreground_times，形状为(frame_h, frame_w)，记录当前帧的每个像素连续多少次判定为前景
    """
    frame_h, frame_w = gray_frame.shape
    background_samples = np.zeros((frame_h, frame_w, N))
    pad_frame = np.pad(gray_frame, (pad_size, pad_size), "symmetric")
    for r in range(frame_h):
        for c in range(frame_w):
            background_sample_of_current_pixel = [gray_frame[r, c]]
            window = pad_frame[r:r + filter_size, c:c + filter_size].ravel()
            background_sample_of_current_pixel.extend(rd.choice(window, N - 1, replace=True))
            background_sample_of_current_pixel = np.array(background_sample_of_current_pixel)
            background_samples[r, c] = background_sample_of_current_pixel
    background_times = np.ones((frame_h, frame_w))
    foreground_times = np.zeros_like(background_times)
    background_samples_keep_times = np.ones((frame_h, frame_w, N))  # 记录背景样本集中每个样本连续没有得到更新的次数
    background_marks = np.ones_like(background_times).astype(np.bool)
    foreground_marks = np.zeros_like(foreground_times).astype(np.bool)
    return background_samples, background_times, foreground_times, background_samples_keep_times, background_marks, foreground_marks


def frame_process(frame, frame_h, frame_w):
    """
    对每一帧图像进行处理，包括resize和灰度化
    :param frame: 帧
    :param frame_h: resize的帧高
    :param frame_w: resize的帧宽
    :return: 处理后的帧
    """
    resized_frame = cv2.resize(frame, (frame_w, frame_h))
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    return gray_frame


def judge_foreground(gray_frame, R, Umin, background_samples, N):
    """
    判断一个像素是否为前景
    :param N: 每个像素的背景样本集容量
    :param frame: 当前像素，形状为(frame_h, frame_w)
    :param R: 球体半径范围
    :param Umin: 判定为背景的最小背景样本数目
    :param background_samples: 所有像素的背景样本集合，形状为(frame_h, frame_w, N)
    :return: foreground_marks，形状为(frame_h, frame_w)，元素为bool类型，True表示当前像素为前景，False表示当前像素为背景
    """
    gray_frame = np.expand_dims(gray_frame, axis=2)  # (frame_h, frame_w, 1)
    gray_frame = np.repeat(gray_frame, N, axis=2)  # (frame_h, frame_w, N)
    foreground_marks = np.logical_not(np.sum(np.abs(gray_frame - background_samples) < R, axis=2) > Umin)  # （frame_h, frame_w），True为foreground像素点，False为background像素点
    return foreground_marks


def update_background_samples(is_first_frame, gray_frame, background_times, foreground_times, background_samples, background_samples_keep_times, last_background_marks, last_foreground_marks):
    """
    更新当前帧的背景样本集合
    :param is_first_frame: True表示第一帧，False表示不是第一帧
    :param gray_frame: resize和灰度化后的帧
    :param background_times: 当前帧前每个像素已经连续为背景的次数
    :param foreground_times: 当前帧前每个像素已经连续为前景的次数
    :param background_samples: 背景样本集合
    :param background_samples_keep_times: 记录背景样本集中每个样本连续没有得到更新的次数
    :return:
    """
    # 初始化背景样本集
    if is_first_frame:
        background_samples, background_times, foreground_times, background_samples_keep_times, background_marks, foreground_marks = init_background_samples(gray_frame, N)  # (frame_h, frame_w, N)
        return background_samples, background_times, foreground_times, background_samples_keep_times, background_marks, foreground_marks
    # 判断前景样本，背景样本点
    foreground_marks = judge_foreground(gray_frame, R, Umin, background_samples, N)
    background_marks = np.logical_not(foreground_marks)
    # 计算背景样本集中元素被保留的概率
    background_samples_keep_prob = calc_background_samples_keep_prob(background_samples_keep_times, N)
    min_keep_prob_index = np.argmin(background_samples_keep_prob, axis=2)
    # 找出能够得到更新的背景像素点的mask
    background_pixel_update_mark = np.logical_and(rd.random(background_times.shape) < pixel_update_prob, background_marks)
    # 更新背景像素点的背景样本集合
    for r, c in zip(*np.where(background_pixel_update_mark)):
        sample_index = min_keep_prob_index[r, c]
        background_samples[r, c, sample_index] = gray_frame[r, c]
        background_samples_keep_times[r, c] += 1
        background_samples_keep_times[r, c, sample_index] = 1
        # TODO:更新邻居
        if rd.random() < pixel_update_prob:
            for r_n in [r - 1, r, r + 1]:
                if r_n < 0 or r_n >= frame_h:
                    continue
                for c_n in [c - 1, c, c + 1]:
                    if c_n < 0 or c_n >= frame_w:
                        continue
                    sample_index = min_keep_prob_index[r_n, c_n]
                    background_samples[r, c, sample_index] = gray_frame[r_n, c_n]
                    background_samples_keep_times[r_n, c_n] += 1
                    background_samples_keep_times[r_n, c_n, sample_index] = 1
                    # break
    # 更新background_times, foreground_times
    background_times[foreground_marks] = 0
    foreground_times[background_marks] = 0
    background_times[background_marks] += 1
    foreground_times[foreground_marks] += 1
    # 判断foreground_times是否有超过foreground_pixel_keep_times的，如果有将其变为背景像素点，并按照概率更新自己的背景样本集
    foreground_to_background_mark = foreground_times > foreground_pixel_keep_times
    foreground_times[foreground_to_background_mark] = 0
    background_times[foreground_to_background_mark] = 1
    foreground_marks[foreground_to_background_mark] = False
    background_marks[foreground_to_background_mark] = True
    update_sample_mark = np.logical_and(rd.random(foreground_times.shape) < pixel_update_prob, foreground_to_background_mark)
    for r, c in zip(*np.where(update_sample_mark)):
        sample_index = min_keep_prob_index[r, c]
        background_samples[r, c, sample_index] = gray_frame[r, c]
        background_samples_keep_times[r, c] += 1
        background_samples_keep_times[r, c, sample_index] = 1
    return background_samples, background_times, foreground_times, background_samples_keep_times, background_marks, foreground_marks


def detect():
    cap = cv2.VideoCapture(0)
    is_first_frame = True
    while cap.isOpened():
        # get a frame
        ret, frame = cap.read()
        original_h, original_w = frame.shape[:2]
        gray_frame = frame_process(frame, frame_h, frame_w)
        if is_first_frame:
            background_samples, background_times, foreground_times, background_samples_keep_times, background_marks, foreground_marks = update_background_samples(is_first_frame, gray_frame, None, None, None, None, None, None)
            is_first_frame = False
            foreground_mask = cv2.resize(foreground_marks.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST) * 255
            frame[foreground_mask == 255] = np.array([0, 0, 255])
            cv2.imshow("capture", frame)
            continue
        background_samples, background_times, foreground_times, background_samples_keep_times, background_marks, foreground_marks = update_background_samples(is_first_frame, gray_frame, background_times, foreground_times, background_samples, background_samples_keep_times, background_marks, foreground_marks)
        # show a frame
        foreground_mask = cv2.resize(foreground_marks.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST) * 255
        frame[foreground_mask == 255] = np.array([0, 0, 255])
        cv2.imshow("capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
   detect()
