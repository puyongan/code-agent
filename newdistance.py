import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import sys
import math
import subprocess
from pathlib import Path

# ========= 0. parameters setting ==========
pic_path='./photos/test.jpg'
param=0.02
# stand=1
star_mode = False


# 输出文件夹路径
output_path = Path("ai_gen")

# 判断文件夹是否存在
if not output_path.exists():
    output_path.mkdir(parents=True)

# ============== 1. 拍照函数 ==============
def capture_and_save(filename=pic_path, camera_id=1):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    ret, frame = cap.read()
    if ret:
        cv2.imwrite(filename, frame)
        print(f"照片已保存为 {filename}")
    else:
        print("拍照失败")

    cap.release()

# 拍照
# capture_and_save()



# ============== 2. 图像预处理 ==============
img = cv2.imread(pic_path)
img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)

# 自适应二值化 + 形态学优化
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # 连接边缘
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)   # 去除噪声

# ========= 3. 轮廓检测与筛选 ========
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 取最大轮廓
contours = sorted(contours, key=cv2.contourArea, reverse=True)
max_contour = contours[0]

# ==== 4. 多边形逼近（提取角点） =======

epsilon = param * cv2.arcLength(max_contour, True)  # 调整系数
approx = cv2.approxPolyDP(max_contour, epsilon, True)

# 暂时不作角点个数审核
stand = len(approx)

if len(approx) != stand:
    print(f"warning: {len(approx)} corners dectected,（{stand}expected）adjust param")
else:
    print(f"{stand}corners detected")
    corners = [tuple(point[0]) for point in approx]

    # 使得最上面那个点为起点，后面按顺时针排列
    # 1. 找到最上面的点作为起点
    start_idx = np.argmin([y for x, y in corners])
    start_point = corners[start_idx]

    # 2. 按顺时针排序
    import math
    def angle_from_start(pt, start):
        x, y = pt
        sx, sy = start
        return math.atan2(y - sy, x - sx)
    other_points = [pt for i, pt in enumerate(corners) if i != start_idx]
    sorted_points = sorted(other_points, key=lambda pt: angle_from_start(pt, start_point))
    corners = [start_point] + sorted_points


    # ============== 五角星模式支持 ==============
    if len(sys.argv) > 1 and sys.argv[1] == '1':
        star_mode = True
        # 通用五角星重排逻辑：隔一个取点，模5除法

        star_order = [(i*2)%stand for i in range(stand)]
        corners = [corners[i] for i in star_order]
        # 连线五角星，绿色线
        for i in range(len(corners)):
            pt1 = corners[i]
            pt2 = corners[(i+1)%stand]
            cv2.line(img, pt1, pt2, (0,255,0), 15)


    # ============== 5. 可视化（用PIL绘制中文） ==============
    result_img = img.copy()
    def draw(img, text, pos, color=(0,0,0)):
        x, y = pos
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 60)
        draw.text((x,y), text, font=font, fill=color)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    for i, (x,y) in enumerate(corners):
        cv2.circle(result_img, (x,y), 20, (0,0,255), -1)
        result_img = draw(result_img, f"corner{i+1}", (x+10, y-10))

    # ============== 6. 保存与显示 ==============
    def normalize_to_canvas(points):
        # 获取照片画布大小
        img_height, img_width = img.shape[:2]
        print(f"照片尺寸：宽度 = {img_width}px，高度 = {img_height}px")  # 新增打印语句

        # # 计算目标画布的长边（用于安全边界）
        target_long_edge = 2000

        # 计算照片的长边
        photo_long_edge = max(img_width, img_height)

        # 计算缩放因子：照片长边 / 目标画布长边（确保安全裕度）
        scale = photo_long_edge / target_long_edge

        return [
            (x / scale, y / scale)
            for x, y in points
        ]

    norm_corners = normalize_to_canvas(corners)
    with open(f"{output_path / 'points.txt'}", 'w') as f:
        for x,y in norm_corners:
            f.write(f"{int(x)} {int(y)}\n")

    # ============== 差值计算并保存 ==============
    with open(f"{output_path / 'zuobiao.txt'}", 'w') as f:
        num_points = len(norm_corners)
        for i in range(num_points-1):
            x1, y1 = norm_corners[i]
            x2, y2 = norm_corners[i+1]
            dx = int(x2 - x1)
            dy = int(y2 - y1)
            if dx < 0:
                dx_str = "1" + str(abs(dx))
            else:
                dx_str = "2" + str(abs(dx))
            if dy < 0:
                dy_str = "1" + str(abs(dy))
            else:
                dy_str = "2" + str(abs(dy))
            f.write(f"{dx_str} {dy_str}\n")
        # 最后一个点到起点
        x1, y1 = norm_corners[-1]
        x2, y2 = norm_corners[0]
        dx = int(x2 - x1)
        dy = int(y2 - y1)
        if dx < 0:
            dx_str = "1" + str(abs(dx))
        else:
            dx_str = "2" + str(abs(dx))
        if dy < 0:
            dy_str = "1" + str(abs(dy))
        else:
            dy_str = "2" + str(abs(dy))
        f.write(f"{dx_str} {dy_str}\n")

    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title("result")
    plt.axis('off')
    plt.show()
    print(f"拐点坐标已保存到 {output_path / 'points.txt' }")
    print(f"差值坐标已保存到 {output_path / 'zuobiao.txt' }")
    # ========== 自动SCP上传 ==========
    # scp_cmd = (
    #     f'Scp {output_path / 'zuobiao.txt'} admin1@192.168.0.100:/home/admin1/Documents/BSI_EContest/algorithm'
    # )
    # try:
    #     subprocess.Popen([
    #         'powershell',
    #         '-Command',
    #         f'Start-Process powershell -ArgumentList \'-NoExit\', \'{scp_cmd}\''
    #     ])
    #     print("已弹出PowerShell窗口，请手动输入密码完成上传。")
    # except Exception as e:
    #     print("打开PowerShell出错：", e)

    # 暂时不进行上传

    # ========== 自动SCP上传（带密码自动输入） ==========
    # # 单行命令上传（密码明文在命令行中）
    # subprocess.run(
    #     f'echo y | pscp -pw admin123 ai_gen/zuobiao.txt admin1@192.168.0.100:"/home/admin1/Documents/BSI_EContest/algorithm/"',
    #     shell=True
    # )
