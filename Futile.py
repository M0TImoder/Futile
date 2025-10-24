import pygame
import sys
import math

pygame.init()
W, H = 1280, 720
scr = pygame.display.set_mode((W, H))
pygame.display.set_caption("Futile - Commented")
clk = pygame.time.Clock()
cx, cy = W // 2, H // 2

# Core param
spd = 180.0        # 基本移動速度（units / sec）
dash_mul = 1.9     # ダッシュ時に速度に掛かる倍率
acc = 15.0         # 速度追従の速さ
fric = 8.0         # 地上でアイドル時に速度が落ちる速さ（簡易摩擦）
m_sens = 0.002     # マウス感度
fov_d = 400.0      # 距離パラメータ（投影用）

# Physics param
grav = -300.0      # 重力（絶対値と強さが比例）
jump_v = 140.0     # ジャンプ初速（jump_v^2/(2*|grav|)）
jump_step = 10.0   # 実行中に +/- で調整するステップ量（デバッグ用）
jump_default = jump_v

# Render param
ground_y_base = -200.0  # 基本地面の高さ（平地）
grid_off = -30.0        # グリッド（表示）を地面からさらに下げる量（視認性調整）
g_size = 100            # グリッドの1セルサイズ
g_range = 900           # グリッドを描画する半径（距離）
eye_h = 40.0            # 目の高さ（プレイヤーの視点は地面 + eye_h）
max_step = 30.0         # 段差耐性（この高さ差より高いと登れない）

# State
cam = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "pitch": 0.0}  # カメラ/プレイヤー状態
vel_y = 0.0         # 垂直速度
vx, vz = 0.0, 0.0   # 水平速度成分（慣性つけるため分離）
on_g = True         # 地面に接地しているかどうか
slopes = []         # 坂情報のリスト（外部から編集する想定）

# Helper function
def clamp(v, a, b):
    """値vを[a,b]に収めるユーティリティ"""
    return max(a, min(b, v))

def proj(px, py, pz):
    """
    透視投影（pzが極端に小さいときに爆発するのを防ぐため）
    """
    pz_clip = max(pz, 0.05)
    f = fov_d / (pz_clip + fov_d)
    return (px * f + cx, -py * f + cy)

def rot_cam(px, py, pz, cam):
    """
    カメラ回転（yaw,pitch）をワールド座標→ビュー座標に適用する簡易関数
    """
    y, p = cam["yaw"], cam["pitch"]
    # yaw
    x = px * math.cos(y) - pz * math.sin(y)
    z = px * math.sin(y) + pz * math.cos(y)
    # pitch
    yy = py * math.cos(p) - z * math.sin(p)
    zz = py * math.sin(p) + z * math.cos(p)
    return x, yy, zz

def g_h(x, z):
    """
    地面の高さを返す関数
    - 基本はground_y_base
    - slopesに登録された局所的な傾斜を合算して返す
    """
    y = ground_y_base
    for s in slopes:
        dx = x - s["x"]; dz = z - s["z"]
        if abs(dx) < s["w"] * 0.5 and abs(dz) < s["d"] * 0.5:
            # そのスロープの領域内なら、x,z に応じた高さを加える
            y += dx * s["sx"] + dz * s["sz"]
    return y

# I/O
def read_input():
    """
    マウス相対移動とキー入力を読み取る
    """
    try:
        mx, my = pygame.mouse.get_rel()
    except Exception:
        mx, my = 0, 0

    # マウスで視点回転（安全にクリップ）
    cam["yaw"] += mx * m_sens
    cam["pitch"] -= my * m_sens
    cam["pitch"] = clamp(cam["pitch"], -math.pi/2 + 0.01, math.pi/2 - 0.01)

    keys = pygame.key.get_pressed()
    # カメラ基準の前/右ベクトル（ワールドXZ平面）
    fx = math.sin(cam["yaw"]); fz = math.cos(cam["yaw"])
    rx = math.cos(cam["yaw"]); rz = -math.sin(cam["yaw"])

    mvx = mvz = 0.0
    if keys[pygame.K_w]: mvx += fx; mvz += fz
    if keys[pygame.K_s]: mvx -= fx; mvz -= fz
    if keys[pygame.K_a]: mvx -= rx; mvz -= rz
    if keys[pygame.K_d]: mvx += rx; mvz += rz

    dash = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
    jump = keys[pygame.K_SPACE]
    return mvx, mvz, dash, jump

# Logic
def handle_movement(mvx, mvz, dash, jump, dt):
    global vx, vz, vel_y, on_g, cam

    mag = math.hypot(mvx, mvz)
    desire_x = desire_z = 0.0
    if mag > 0.001:
        desire_x = mvx / mag
        desire_z = mvz / mag

    # ジャンプ処理は必ず先に扱う（停止時でもジャンプ可能にするため）
    if jump and on_g:
        vel_y = jump_v
        on_g = False

    # ターゲット速度（ダッシュ時は倍率を掛ける）
    target_speed = spd * (dash_mul if dash else 1.0)
    tar_vx = desire_x * target_speed
    tar_vz = desire_z * target_speed

    # 加速（1次系）
    vx += (tar_vx - vx) * clamp(acc * dt, 0.0, 1.0)
    vz += (tar_vz - vz) * clamp(acc * dt, 0.0, 1.0)

    # 空中では横制御が弱まる
    if not on_g:
        vx *= (1.0 - clamp(6.0 * dt, 0.0, 0.8))
        vz *= (1.0 - clamp(6.0 * dt, 0.0, 0.8))

    # 移動量（フレーム時間に合わせてスケーリング）
    dx = vx * dt
    dz = vz * dt

    # 移動量が無視できるなら摩擦をかけて抜ける（早期抜け）
    if abs(dx) < 1e-6 and abs(dz) < 1e-6:
        if on_g:
            vx -= vx * clamp(fric * dt, 0.0, 1.0)
            vz -= vz * clamp(fric * dt, 0.0, 1.0)
        return

    # 現在位置と移動先の地面高さを比較して段差判定
    cur_gy = g_h(cam["x"], cam["z"])
    nx = cam["x"] + dx
    nz = cam["z"] + dz
    nxt_gy = g_h(nx, nz)
    step = nxt_gy - cur_gy  # 正なら次の位置が高い

    if on_g:
        if step <= max_step:
            # 許容段差なら移動して高さをスナップ（即座に目の高さを合わせる）
            cam["x"], cam["z"] = nx, nz
            cam["y"] = nxt_gy + eye_h
        else:
            # 急すぎると進めない
            # 摩擦モデルへ変更予定
            if mag > 0.001:
                # 簡易的な衝突判定
                dot = vx * desire_x + vz * desire_z
                if dot > 0:
                    vx -= dot * desire_x
                    vz -= dot * desire_z
            # 速度を少し減衰させて暴れを止める
            vx *= 0.6; vz *= 0.6
    else:
        # 空中ならそのまま移動
        cam["x"], cam["z"] = nx, nz

# Vertical physics
def upd_phys(dt):
    """
    - vel_y に重力を積分
    - cam["y"] に垂直移動を適用
    - 現在の x,z に対応する地面高さに到達したら着地（位置スナップ）
    """
    global vel_y, on_g, cam
    vel_y += grav * dt
    cam["y"] += vel_y * dt

    base_h = g_h(cam["x"], cam["z"]) + eye_h
    if cam["y"] <= base_h:
        cam["y"] = base_h
        vel_y = 0.0
        on_g = True
    else:
        on_g = False

# Gulid
def draw_grid():
    """
    カメラ周囲のセルを走査して四隅を投影して線を引く
    """
    # near plane を使ってクリップ→三角分解する予定
    sx = int(cam["x"] // g_size) * g_size - g_range
    ex = int(cam["x"] // g_size) * g_size + g_range
    sz = int(cam["z"] // g_size) * g_size - g_range
    ez = int(cam["z"] // g_size) * g_size + g_range
    col = (70,70,70)
    inner_r = 200  # 内側は強制描画（単位: world distance）

    for x in range(sx, ex + g_size, g_size):
        for z in range(sz, ez + g_size, g_size):
            dxp = x - cam["x"]; dzp = z - cam["z"]
            dist = math.hypot(dxp, dzp)
            if dist > g_range: 
                continue

            # 各セルの四隅の高さを地形に従って取得
            y1 = g_h(x, z) + grid_off
            y2 = g_h(x + g_size, z) + grid_off
            y3 = g_h(x + g_size, z + g_size) + grid_off
            y4 = g_h(x, z + g_size) + grid_off

            pts = []
            for vx_, vy_, vz_ in ((x,y1,z),(x+g_size,y2,z),(x+g_size,y3,z+g_size),(x,y4,z+g_size)):
                px = vx_ - cam["x"]; py = vy_ - cam["y"]; pz = vz_ - cam["z"]
                rx, ry, rz = rot_cam(px, py, pz, cam)
                # 内側では強制的に少し前にある扱いにして投影（消えないためのトリック）
                if dist < inner_r and rz <= 0.05:
                    rz = 0.05
                if rz > 0.05:
                    pts.append(proj(rx, ry, rz))
            if len(pts) == 4:
                pygame.draw.lines(scr, col, True, pts, 1)

# debug
def draw_debug(fps):
    font = pygame.font.SysFont("consolas", 18)
    info = (f"pos: ({cam['x']:.1f},{cam['y']:.1f},{cam['z']:.1f}) "
            f"yaw:{math.degrees(cam['yaw']):.1f} pitch:{math.degrees(cam['pitch']):.1f} "
            f"on_g:{on_g} vx:{vx:.1f} vz:{vz:.1f} fps:{int(fps)} "
            f"slopes:{len(slopes)} jump_v:{jump_v:.1f} grav:{grav:.1f}")
    scr.blit(font.render(info, True, (200,200,200)), (10,10))

# init
cam["y"] = g_h(cam["x"], cam["z"]) + eye_h
pygame.event.set_grab(True)
pygame.mouse.set_visible(False)

# Main loop
while True:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            pygame.quit(); sys.exit()

        # Escape で即終了（素敵）
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                pygame.quit(); sys.exit()
            # 実行中にジャンプ力を微調整するデバッグ用キー
            if e.key == pygame.K_LEFTBRACKET:   # [
                jump_v = max(0.0, jump_v - jump_step)
            if e.key == pygame.K_RIGHTBRACKET:  # ]
                jump_v = jump_v + jump_step
            if e.key == pygame.K_0:
                jump_v = jump_default

        # ウィンドウフォーカス喪失時の安全措置（mouse.get_rel の暴発対策）
        if e.type == pygame.ACTIVEEVENT:
            try:
                if e.state & 2 and not e.gain:
                    pygame.mouse.get_rel()  # deltaを吐き出して次回へ備える
            except Exception:
                pass

    # dt（秒）で時間を扱う
    ms = clk.tick(60)
    dt = ms / 1000.0
    fps = clk.get_fps()

    # 入力から描画までのパイプライン
    mvx, mvz, dash, jump = read_input()
    handle_movement(mvx, mvz, dash, jump, dt)
    upd_phys(dt)

    scr.fill((15,15,20))
    draw_grid()
    draw_debug(fps)
    pygame.display.flip()
