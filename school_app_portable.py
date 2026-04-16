# %% [Cell 1: 全局配置]
import sys
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QFormLayout, QLineEdit, QPushButton, 
                             QLabel, QComboBox, QGroupBox, QScrollArea, QMessageBox,
                             QSlider, QCheckBox, QInputDialog, QGridLayout, QFileDialog) 
from PySide6.QtCore import Qt
from PySide6.QtGui import QDoubleValidator, QIntValidator
import xgboost
import lightgbm


def configure_matplotlib_chinese_font():
    """优先选择系统中可用的中文字体，避免中文显示为方框。"""
    candidates = [
        "Microsoft YaHei", "SimHei", "SimSun", "KaiTi", "FangSong",
        "Noto Sans CJK SC", "Source Han Sans SC", "Arial Unicode MS"
    ]
    installed_fonts = {font.name for font in fm.fontManager.ttflist}

    for font_name in candidates:
        if font_name in installed_fonts:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return font_name

    plt.rcParams["axes.unicode_minus"] = False
    return None


MATPLOTLIB_CJK_FONT = configure_matplotlib_chinese_font()

# ================== 1. 全局配置 ==================
def get_base_path():
    if getattr(sys, "frozen", False):
        return os.path.dirname(getattr(sys, "_MEIPASS", os.path.dirname(sys.executable)))
    return os.path.dirname(os.path.abspath(__file__))


BASE_PATH = get_base_path()
TARGET_KEYS = ["EUI", "sDA", "sGA", "UDI", "SVF"]
TARGET_DISPLAY = {
    "EUI": "EUI (kWh/m²)",
    "sDA": "sDA<sub>450/50%</sub> (%)",
    "sGA": "sGA<sub>0.35, 5%</sub> (%)",
    "UDI": "UDI<sub>100-2000lx</sub> (%)",
    "SVF": "SVF (%)"
}

FEATURE_ORDER = [
    'WWR_s_exp', 'WWR_s', 'H_s', 'W_s', 'SH_s', 'd_A_s', 'd_B_s',
    'WWR_c', 'H_c', 'W_c', 'SH_c', 'd_A_c', 'd_B_c',
    'F_BL', 'F_BR', 'F_TR', 'F_TL',
    'alpha_oh', 'L_oh', 'd_mv',
    'N_v', 'W_v', 'L_v', 'M_TL', 'M_TR'
]

FEATURE_META = {
    'WWR_s_exp': {'symbol': 'WWRs',  'label': '外窗侧窗墙比', 'unit': '%',  'min': 30,   'max': 70,   'step': 5},
    'WWR_s':     {'symbol': 'WWRs',  'label': '外窗侧窗墙比', 'unit': '%',  'min': 30,   'max': 70,   'step': 5},
    'H_s':       {'symbol': 'Hs',    'label': '窗高',         'unit': 'm',  'min': 1.2,  'max': 2.6,  'step': 0.1},
    'W_s':       {'symbol': 'Ws',    'label': '窗宽',         'unit': 'm',  'min': 4.0,  'max': 7.8,  'step': 0.1},
    'SH_s':      {'symbol': 'SHs',   'label': '窗台高度',     'unit': 'm',  'min': 0.4,  'max': 1.0,  'step': 0.1},
    'd_A_s':     {'symbol': 'dA,s',  'label': '采光窗左距',   'unit': 'm',  'min': 1.0,  'max': 8.0,  'step': 0.1},
    'd_B_s':     {'symbol': 'dB,s',  'label': '采光窗右距',   'unit': 'm',  'min': 0.2,  'max': 8.0,  'step': 0.1},
    'WWR_c':     {'symbol': 'WWRc',  'label': '走廊侧窗墙比', 'unit': '%',  'min': 10,   'max': 40,   'step': 5},
    'H_c':       {'symbol': 'Hc',    'label': '窗高',         'unit': 'm',  'min': 0.6,  'max': 2.0,  'step': 0.1},
    'W_c':       {'symbol': 'Wc',    'label': '窗宽',         'unit': 'm',  'min': 3.4,  'max': 6.0,  'step': 0.1},
    'SH_c':      {'symbol': 'SHc',   'label': '窗台高度',     'unit': 'm',  'min': 0.8,  'max': 2.0,  'step': 0.1},
    'd_A_c':     {'symbol': 'dA,c',  'label': '走廊窗左距',   'unit': 'm',  'min': 0.1,  'max': 6.0,  'step': 0.1},
    'd_B_c':     {'symbol': 'dB,c',  'label': '走廊窗右距',   'unit': 'm',  'min': 0.1,  'max': 6.0,  'step': 0.1},
    'F_BL':      {'symbol': 'FBL',   'label': '框底左宽度',   'unit': 'm',  'min': 0.1,  'max': 1.5,  'step': 0.1},
    'F_BR':      {'symbol': 'FBR',   'label': '框底右宽度',   'unit': 'm',  'min': 0.1,  'max': 1.5,  'step': 0.1},
    'F_TR':      {'symbol': 'FTR',   'label': '框顶右宽度',   'unit': 'm',  'min': 0.1,  'max': 1.5,  'step': 0.1},
    'F_TL':      {'symbol': 'FTL',   'label': '框顶左宽度',   'unit': 'm',  'min': 0.1,  'max': 1.5,  'step': 0.1},
    'alpha_oh':  {'symbol': 'αoh',   'label': '悬挑角度',     'unit': '°',  'min': -80,  'max': 80,   'step': 1},
    'L_oh':      {'symbol': 'Loh',   'label': '悬挑长度',     'unit': 'm',  'min': 0.1,  'max': 2.0,  'step': 0.1},
    'd_mv':      {'symbol': 'dmv',   'label': '悬挑移动距离', 'unit': 'm',  'min': 0.0,  'max': 3.8,  'step': 0.1},
    'N_v':       {'symbol': 'Nv',    'label': '垂直遮阳数量', 'unit': '片', 'min': 1,    'max': 10,   'step': 1},
    'W_v':       {'symbol': 'Wv',    'label': '垂直遮阳宽度', 'unit': 'm',  'min': 0.01, 'max': 1.2,  'step': 0.05,
                  'slider_values': [0.01] + [round(i * 0.05, 2) for i in range(1, 25)]},
    'L_v':       {'symbol': 'Lv',    'label': '垂直遮阳长度', 'unit': 'm',  'min': 0.1,  'max': 1.5,  'step': 0.1},
    'M_TL':      {'symbol': 'MTL',   'label': '左上移动距离', 'unit': 'm',  'min': 0.0,  'max': 1.2,  'step': 0.05},
    'M_TR':      {'symbol': 'MTR',   'label': '右上移动距离', 'unit': 'm',  'min': 0.0,  'max': 1.2,  'step': 0.05},
}

SHADE_FEATURE_MAP = {
    "基准 (Base)": FEATURE_ORDER[1:13],
    "框式 (Frame)": FEATURE_ORDER[1:13] + FEATURE_ORDER[13:17],
    "悬挑 (Overhang)": FEATURE_ORDER[1:13] + FEATURE_ORDER[17:20],
    "垂直 (Vertical)": FEATURE_ORDER[1:13] + FEATURE_ORDER[20:25],
    "组合 (O+V)": FEATURE_ORDER[1:13] + FEATURE_ORDER[17:25]
}

# 部分早期模型训练时使用了不同列名，此映射将训练列名解析为代码内部统一名称
FEATURE_ALIAS = {
    'Frame_BL': 'F_BL',
    'Frame_BR': 'F_BR',
    'Frame_TR': 'F_TR',
    'Frame_TL': 'F_TL',
}

ORI_LIST = ["南向 (South 0°)", "北向 (North 0°)"]
SHADE_LIST = ["基准 (Base)", "悬挑 (Overhang)", "垂直 (Vertical)", "组合 (O+V)", "框式 (Frame)"]

PRESET_DATA = {
    (ORI_LIST[0], SHADE_LIST[0]): {'WWR_s':30,'H_s':2,'SH_s':0.9,'WWR_c':15,'H_c':0.9,'W_c':4.5,'SH_c':0.8},
    (ORI_LIST[0], SHADE_LIST[1]): {'WWR_s':30,'H_s':1.2,'SH_s':1,'WWR_c':15,'H_c':1.2,'W_c':3.4,'SH_c':0.8,'alpha_oh':20,'L_oh':1.9,'d_mv':0},
    (ORI_LIST[0], SHADE_LIST[2]): {'WWR_s_exp':31,'H_s':2.6,'SH_s':0.4,'WWR_c':10,'H_c':0.8,'W_c':3.4,'SH_c':2,'N_v':2,'W_v':1,'L_v':0.1,'M_TL':0,'M_TR':0.2},
    (ORI_LIST[0], SHADE_LIST[3]): {'WWR_s_exp':31,'H_s':1.4,'SH_s':0.8,'WWR_c':20,'H_c':0.9,'W_c':6,'SH_c':0.8,'alpha_oh':50,'L_oh':2,'d_mv':0,'N_v':7,'W_v':0.3,'L_v':0.2,'M_TL':0.15,'M_TR':0},
    (ORI_LIST[0], SHADE_LIST[4]): {'WWR_s':30,'H_s':2,'SH_s':1,'WWR_c':15,'H_c':1,'W_c':4,'SH_c':1,'F_BL':0.7,'F_BR':1.4,'F_TR':1.1,'F_TL':0.9},
    (ORI_LIST[1], SHADE_LIST[0]): {'WWR_s':40,'H_s':2.6,'SH_s':0.4,'WWR_c':10,'H_c':0.8,'W_c':3.4,'SH_c':2},
    (ORI_LIST[1], SHADE_LIST[1]): {'WWR_s':35,'H_s':2.2,'SH_s':0.8,'WWR_c':15,'H_c':0.9,'W_c':4.5,'SH_c':0.8,'alpha_oh':80,'L_oh':0.4,'d_mv':0.8},
    (ORI_LIST[1], SHADE_LIST[2]): {'WWR_s_exp':40,'H_s':1.9,'SH_s':1,'WWR_c':10,'H_c':0.8,'W_c':3.4,'SH_c':2,'N_v':8,'W_v':0.3,'L_v':0.1,'M_TL':0.1,'M_TR':0},
    (ORI_LIST[1], SHADE_LIST[3]): {'WWR_s_exp':40,'H_s':2.6,'SH_s':0.4,'WWR_c':10,'H_c':0.6,'W_c':4.5,'SH_c':2,'alpha_oh':-80,'L_oh':0.9,'d_mv':0.5,'N_v':5,'W_v':0.02,'L_v':0.1,'M_TL':0,'M_TR':0},
    (ORI_LIST[1], SHADE_LIST[4]): {'WWR_s':40,'H_s':2.6,'SH_s':0.4,'WWR_c':10,'H_c':0.8,'W_c':3.4,'SH_c':2,'F_BL':0.1,'F_BR':0.4,'F_TR':0.1,'F_TL':0.1}
}


# %% [Cell 2: 几何计算引擎]
def geometry_engine(ui, mode):
    wa = 27.0  # 墙面积
    wl = 9.0   # 墙长
    wall_h = 3.8
    min_top_clearance = 0.8
    
    def safe_float(key, default=0.0):
        val = ui.get(key, default)
        try:
            return float(val) if val is not None else default
        except:
            return default

    def skylight_height_cap(wwr_value):
        if abs(wwr_value - 30.0) < 1e-9:
            return 2.0
        if abs(wwr_value - 35.0) < 1e-9:
            return 2.2
        return 2.6

    def adjust_skylight_sill(hs_value, shs_value):
        shs_actual = min(max(round(shs_value, 2), 0.4), 1.0)
        top_gap = round(wall_h - hs_value - shs_actual, 2)

        while top_gap < min_top_clearance and shs_actual > 0.4:
            shs_actual = round(max(0.4, shs_actual - 0.1), 2)
            top_gap = round(wall_h - hs_value - shs_actual, 2)

        if top_gap < 0.05:
            top_gap = 0.0
        return round(shs_actual, 2), round(top_gap, 2)

    hs_in = safe_float('H_s', 2.0)
    hc = safe_float('H_c', 1.0)
    shs_in = safe_float('SH_s', 0.9)
    shc_in = safe_float('SH_c', 0.8)
    wv = safe_float('W_v', 0.1)
    
    # --- A. 遮阳板物理约束 ---
    L_orig = safe_float('L_oh', 1.0)
    d_orig = safe_float('d_mv', 0.5)
    angle = safe_float('alpha_oh', 0.0)
    
    L_final = L_orig
    d_final = d_orig

    # 当 alpha_oh < 0 时，遮阳板长度 L_oh 必须 <= d_mv + SH_s
    if angle < 0:
        max_L_oh = round(d_final + shs_in, 2)
        if L_final > max_L_oh:
            L_final = max_L_oh
            if L_final < 0:
                L_final = 0

    # --- B. 走廊窗 WWR_c 与 SH_c 联动逻辑 ---
    wwr_c_in = safe_float('WWR_c', 15)
    max_area_c = 6.2 * hc
    max_wwr_c = (max_area_c / wa) * 100.0
    real_wwr_c = min(wwr_c_in, max_wwr_c)
    wc_actual = min((real_wwr_c / 100.0 * wa) / max(0.1, hc), 6.2)

    # H_c <= 0.8 时，SH_c 固定为 2.0；否则限制在 0.8-1.0 之间。
    if hc <= 0.8:
        shc_actual = 2.0
    else:
        shc_actual = min(max(shc_in, 0.8), 1.0)

    # --- C. 南向采光窗联动逻辑 ---
    is_vertical_mode = mode in ["垂直 (Vertical)", "组合 (O+V)"]
    design_wwr_s = safe_float('WWR_s_exp', 30) if is_vertical_mode else safe_float('WWR_s', 30)
    hs_cap = skylight_height_cap(design_wwr_s)
    hs_actual = min(max(round(hs_in, 1), 1.2), hs_cap)
    target_area_s = round((design_wwr_s / 100.0) * wa, 3)

    total_opening_w = round(target_area_s / max(hs_actual, 0.1), 3)
    while total_opening_w >= wl and hs_actual < hs_cap:
        hs_actual = round(min(hs_actual + 0.1, hs_cap), 1)
        total_opening_w = round(target_area_s / max(hs_actual, 0.1), 3)

    total_opening_w = min(total_opening_w, wl)

    if is_vertical_mode:
        nv = max(2, int(safe_float('N_v', 2)))
        net_glass_w_theory = max(0.1, total_opening_w - (nv - 1) * wv)
        ws_pane = round(net_glass_w_theory / max(1, nv - 1), 1)
        net_glass_w_actual = ws_pane * (nv - 1)
        actual_wwr_s = round((net_glass_w_actual * hs_actual / wa) * 100.0)
        total_opening_w_actual = net_glass_w_actual + (nv - 1) * wv
    else:
        nv = 0
        ws_pane = round(total_opening_w, 1)
        actual_wwr_s = round((ws_pane * hs_actual / wa) * 100.0)
        total_opening_w_actual = ws_pane

    shs_actual, _ = adjust_skylight_sill(hs_actual, shs_in)

    # 保持可见构图与参数一致：采光窗默认居中，因此左右距离各占一半。
    remain_w_s = max(0.0, round(wl - total_opening_w_actual, 2))
    d_a_s_actual = round(remain_w_s / 2.0, 2)
    d_b_s_actual = round(remain_w_s - d_a_s_actual, 2)

    # --- D. M_TL + M_TR ≤ W_v 约束 ---
    mtl = safe_float('M_TL', 0.0)
    mtr = safe_float('M_TR', 0.0)
    if mtl + mtr > wv:
        ratio = wv / max(mtl + mtr, 1e-9)
        mtl = round(mtl * ratio, 2)
        mtr = round(mtr * ratio, 2)

    # --- E. 封送结果 ---
    res = {k: safe_float(k) for k in FEATURE_ORDER}
    res.update({
        'H_s': hs_actual,
        'WWR_s': int(actual_wwr_s),
        'WWR_c': int(real_wwr_c),
        'W_s': ws_pane,
        'W_c': round(wc_actual, 2),
        'SH_s': shs_actual,
        'SH_c': round(shc_actual, 2),
        'd_A_s': d_a_s_actual,
        'd_B_s': d_b_s_actual,
        'N_v': nv,
        'M_TL': mtl,
        'M_TR': mtr,
        'L_oh': L_final,
        'd_mv': d_final
    })
    return res


# %% [Cell 3: 3D 物理绘图引擎]
def draw_classroom_3d(ax, data, sha_mode, show_upper=False, show_side=False, facade_filter=None):
    ax.clear()
    L, D, H_total = 9.0, 9.0, 3.8 

    ax.set_proj_type('ortho') 
    span = 14                 
    ax.set_axis_off()
    hide_corridor = facade_filter == 'daylight_front'
    hide_daylight = facade_filter == 'corridor_back'

    # --- 内部函数：绘制单个模块 ---
    def render_module(dy=0, dz=0, is_main=True):
        fac_boost = facade_filter is not None and is_main
        alpha_wall = 0.4 if is_main else 0.1
        alpha_solid = 0.8 if is_main else 0.15
        alpha_glass = 0.5 if is_main else 0.15
        if fac_boost:
            alpha_wall = min(0.85, alpha_wall + 0.35)
            alpha_solid = min(0.98, alpha_solid + 0.12)
            alpha_glass = min(0.9, alpha_glass + 0.38)
        lw_main = 1.5 if is_main else 0.5
        c_edge = 'k' if is_main else 'gray'

        # 1. 绘制主体框架
        for z in [0, H_total]:
            ax.plot3D([0,9,9,0,0], [0+dy,0+dy,9+dy,9+dy,0+dy], [z+dz,z+dz,z+dz,z+dz,z+dz], c_edge, lw=lw_main)
        for x, y in [(0,0),(9,0),(9,9),(0,9)]:
            ax.plot3D([x,x], [y+dy,y+dy], [0+dz,H_total+dz], c_edge, lw=1, alpha=alpha_wall)

        # 2–4. 走廊地坪、吊顶、走廊窗、门（前视图时隐藏，避免半透明叠影）
        if not hide_corridor:
            ax.add_collection3d(Poly3DCollection([[(-3, 0+dy, 0+dz), (0, 0+dy, 0+dz), (0, 9+dy, 0+dz), (-3, 9+dy, 0+dz)]], facecolors='gray', alpha=0.1))
            c_plenum = [[(-3, 0+dy, 3.0+dz), (0, 0+dy, 3.0+dz), (0, 9+dy, 3.0+dz), (-3, 9+dy, 3.0+dz)],
                        [(-3, 0+dy, 3.8+dz), (0, 0+dy, 3.8+dz), (0, 9+dy, 3.8+dz), (-3, 9+dy, 3.8+dz)]]
            ax.add_collection3d(Poly3DCollection(c_plenum, facecolors='gray', alpha=0.3))
            wc_draw, hc_draw, shc_draw = data.get('W_c', 4.0), data.get('H_c', 1.0), data.get('SH_c', 1.0)
            yc_start = (9.0 - wc_draw) / 2
            win_c_verts = [[(0, yc_start+dy, shc_draw+dz), (0, yc_start+wc_draw+dy, shc_draw+dz), 
                            (0, yc_start+wc_draw+dy, shc_draw+hc_draw+dz), (0, yc_start+dy, shc_draw+hc_draw+dz)]]
            ax.add_collection3d(Poly3DCollection(win_c_verts, facecolors='lightgreen', edgecolors='g', alpha=alpha_glass, lw=1))
            for yd in [0.2, 9-0.2-1.2]:
                dr = [[(0, yd+dy, 0+dz), (0, yd+1.2+dy, 0+dz), (0, yd+1.2+dy, 2.2+dz), (0, yd+dy, 2.2+dz)]]
                ax.add_collection3d(Poly3DCollection(dr, facecolors='saddlebrown', edgecolors=c_edge, alpha=alpha_solid))

        # 5. 教室前墙黑板（靠近坐标轴的一侧短墙）
        board_y = 0.02 + dy
        board_x1, board_x2 = 2.0, 7.0
        board_z1, board_z2 = 0.9, 2.3
        board_verts = [[
            (board_x1, board_y, board_z1 + dz), (board_x2, board_y, board_z1 + dz),
            (board_x2, board_y, board_z2 + dz), (board_x1, board_y, board_z2 + dz)
        ]]
        ax.add_collection3d(Poly3DCollection(
            board_verts,
            facecolors='#243b2f',
            edgecolors='black',
            alpha=0.9 if is_main else 0.2,
            lw=1
        ))
        if is_main:
            ax.text((board_x1 + board_x2) / 2, board_y, board_z2 + 0.15 + dz, '黑板',
                    color='black', ha='center', va='bottom', fontsize=10)

        # 6–8. 采光窗、遮阳（后视图时隐藏 x=9 侧，避免叠影）
        is_v = "垂直" in sha_mode or "组合" in sha_mode
        nv = int(data.get('N_v', 0))
        wv, ws_pane, hs, shs = data.get('W_v', 0.1), data.get('W_s', 4.0), data.get('H_s', 2.0), data.get('SH_s', 0.9)
        total_opening_w = ((nv - 1) * ws_pane) + (nv - 1) * wv if (is_v and nv >= 2) else ws_pane
        ys_start = max(1.0, (D - total_opening_w) / 2)
        if ys_start + total_opening_w > 8.9: ys_start = max(1.0, 8.9 - total_opening_w)
        if not hide_daylight:
            num_p = (nv - 1) if (is_v and nv >= 2) else 1
            for i in range(num_p):
                curr_y = ys_start + wv/2 + i * (ws_pane + wv) if (is_v and nv >= 2) else ys_start
                pane = [[(9, curr_y+dy, shs+dz), (9, curr_y+ws_pane+dy, shs+dz), (9, curr_y+ws_pane+dy, shs+hs+dz), (9, curr_y+dy, shs+hs+dz)]]
                ax.add_collection3d(Poly3DCollection(pane, facecolors='skyblue', edgecolors='b', alpha=alpha_glass))
            if is_v and nv > 0:
                lv, mtl, mtr = data.get('L_v', 0), data.get('M_TL', 0), data.get('M_TR', 0)
                for i in range(nv):
                    by = ys_start + i * (ws_pane + wv)
                    p1, p2, p3, p4 = (9, by-wv/2+dy, 0+dz), (9, by+wv/2+dy, 0+dz), (9, by+wv/2+dy, 3.8+dz), (9, by-wv/2+dy, 3.8+dz)
                    p5, p6, p7, p8 = (9+lv, by-wv/2+mtl+dy, 0+dz), (9+lv, by+wv/2-mtr+dy, 0+dz), (9+lv, by+wv/2-mtr+dy, 3.8+dz), (9+lv, by-wv/2+mtl+dy, 3.8+dz)
                    ax.add_collection3d(Poly3DCollection([[p1,p2,p3,p4],[p5,p6,p7,p8],[p1,p5,p8,p4],[p2,p6,p7,p3],[p1,p2,p6,p5],[p4,p3,p7,p8]], facecolors='teal', alpha=alpha_solid, edgecolors='k', lw=0.3))
            if ("悬挑" in sha_mode or "组合" in sha_mode) and data.get('L_oh', 0) > 0:
                alpha, loh, dmv = np.radians(data.get('alpha_oh', 0)), data['L_oh'], data.get('d_mv', 0)
                zt, z_end, x_ext = 3.8 - dmv, (3.8 - dmv) - loh * np.sin(alpha), 9 + loh * np.cos(alpha)
                oh_verts = [[(9, 0+dy, zt+dz), (x_ext, 0+dy, z_end+dz), (x_ext, 9+dy, z_end+dz), (9, 9+dy, zt+dz)]]
                ax.add_collection3d(Poly3DCollection(oh_verts, facecolors='indianred', alpha=alpha_solid, edgecolors='darkred', lw=0.5))
            if "框式" in sha_mode:
                fbl, fbr, ftr, ftl = data.get('F_BL',0), data.get('F_BR',0), data.get('F_TR',0), data.get('F_TL',0)
                p_bl, p_br = (9, ys_start+dy, shs+dz), (9, ys_start+total_opening_w+dy, shs+dz)
                p_tr, p_tl = (9, ys_start+total_opening_w+dy, shs+hs+dz), (9, ys_start+dy, shs+hs+dz)
                e_bl, e_br = (9+fbl, ys_start+dy, shs+dz), (9+fbr, ys_start+total_opening_w+dy, shs+dz)
                e_tr, e_tl = (9+ftr, ys_start+total_opening_w+dy, shs+hs+dz), (9+ftl, ys_start+dy, shs+hs+dz)
                ax.add_collection3d(Poly3DCollection([[p_bl, p_br, e_br, e_bl], [p_br, p_tr, e_tr, e_br], [p_tr, p_tl, e_tl, e_tr], [p_tl, p_bl, e_bl, e_tl]], facecolors='orange', alpha=alpha_glass, edgecolors='darkorange'))
            if is_main: ax.text(9, 4.5+dy, 4.1+dz, f"Actual WWR: {data['WWR_s']}%", color='darkred', fontweight='bold', ha='center')

    # --- 执行集群绘制 ---
    render_module(dy=0, dz=0, is_main=True)
    if show_upper: render_module(dy=0, dz=H_total, is_main=False)
    if show_side:  render_module(dy=D, dz=0, is_main=False)

    # 动态调整视野防止裁切
    ax.set_ylim(-2.5, 20.5 if show_side else 11.5)
    ax.set_zlim(0, 9 if show_upper else 5)
    ax.set_box_aspect((span, span * (1.5 if show_side else 1), H_total * (2 if show_upper else 1)))

    # 右侧图例（与立面裁剪一致，避免列出已隐藏的构件）
    legend_handles = []
    if not hide_corridor:
        legend_handles.extend([
            Patch(facecolor='gray', edgecolor='gray', alpha=0.3, label='走廊侧地坪/吊顶'),
            Patch(facecolor='lightgreen', edgecolor='g', alpha=0.5, label='走廊窗'),
            Patch(facecolor='saddlebrown', edgecolor='black', alpha=0.8, label='门'),
        ])
    if not hide_daylight:
        legend_handles.append(Patch(facecolor='skyblue', edgecolor='b', alpha=0.5, label='采光窗'))
    legend_handles.append(Patch(facecolor='#243b2f', edgecolor='black', alpha=0.9, label='黑板'))
    if not hide_daylight:
        if "垂直" in sha_mode or "组合" in sha_mode:
            legend_handles.append(
                Patch(facecolor='teal', edgecolor='black', alpha=0.8, label='垂直遮阳')
            )
        if "悬挑" in sha_mode or "组合" in sha_mode:
            legend_handles.append(
                Patch(facecolor='indianred', edgecolor='darkred', alpha=0.8, label='悬挑遮阳')
            )
        if "框式" in sha_mode:
            legend_handles.append(
                Patch(facecolor='orange', edgecolor='darkorange', alpha=0.5, label='框式遮阳')
            )

    ax.figure.subplots_adjust(right=0.8)
    ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.02, 0.5),
              frameon=True, fontsize=9, title='颜色说明', title_fontsize=10)


# %% [Cell 4B: GUI 增强 - 方案保存与对比]
class ClassroomPredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EduSync-SZ：深圳中小学校教室能耗与采光性能智能预测系统")
        self.resize(1500, 950)

        self.model_map = {}
        self.model_cache = {}
        self.inputs = {}
        self.rows = {}
        self.saved_schemes = []
        self.last_prediction = None
        self.compare_labels = {}

        self.scan_folders()
        self.init_ui()
        self.apply_presets()

    def scan_folders(self):
        if not os.path.exists(BASE_PATH):
            return

        folders = [d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d))]
        for f in folders:
            ori = ORI_LIST[0] if "South0°" in f else (ORI_LIST[1] if "North0°" in f else None)
            if not ori:
                continue

            sha = SHADE_LIST[0]
            if "Overhang+Vertical" in f:
                sha = SHADE_LIST[3]
            elif "Overhang" in f:
                sha = SHADE_LIST[1]
            elif "Vertical" in f:
                sha = SHADE_LIST[2]
            elif "Frame" in f:
                sha = SHADE_LIST[4]
            self.model_map[(ori, sha)] = os.path.join(BASE_PATH, f)

        preferred_folder = "0312_2000_North0°_Overhang+Vertical(1)_replaced"
        preferred_path = os.path.join(BASE_PATH, preferred_folder)
        if os.path.isdir(preferred_path):
            self.model_map[(ORI_LIST[1], SHADE_LIST[3])] = preferred_path

    def get_feature_value(self, feat):
        w = self.inputs[feat]
        meta = FEATURE_META.get(feat, {})
        min_v = meta.get('min')
        max_v = meta.get('max')

        if isinstance(w, QSlider):
            return float(w.value())

        raw_text = w.text().strip()
        if not raw_text:
            raw_val = float(min_v) if min_v is not None else 0.0
        else:
            try:
                raw_val = float(raw_text)
            except ValueError:
                raw_val = float(min_v) if min_v is not None else 0.0

        if min_v is not None:
            raw_val = max(float(min_v), raw_val)
        if max_v is not None:
            raw_val = min(float(max_v), raw_val)

        if meta.get('type') == 'int':
            raw_val = int(round(raw_val))
        else:
            decimals = int(meta.get('decimals', 2))
            raw_val = round(raw_val, decimals)

        return float(raw_val)

    def invalidate_prediction(self, *args):
        self.last_prediction = None
        if hasattr(self, 'btn_save_scheme'):
            self.btn_save_scheme.setEnabled(False)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        scroll = QScrollArea()
        scroll.setFixedWidth(430)
        scroll.setWidgetResizable(True)
        left_container = QWidget()
        vbox = QVBoxLayout(left_container)

        grp_sel = QGroupBox("1. 策略选择")
        f_sel = QFormLayout(grp_sel)
        self.cb_ori, self.cb_sha = QComboBox(), QComboBox()
        self.cb_ori.addItems(ORI_LIST)
        self.cb_sha.addItems(SHADE_LIST)
        self.cb_ori.currentTextChanged.connect(self.apply_presets)
        self.cb_sha.currentTextChanged.connect(self.apply_presets)
        self.cb_ori.currentTextChanged.connect(self.invalidate_prediction)
        self.cb_sha.currentTextChanged.connect(self.invalidate_prediction)
        f_sel.addRow("朝向:", self.cb_ori)
        f_sel.addRow("形式:", self.cb_sha)
        vbox.addWidget(grp_sel)

        grp_param = QGroupBox("2. 设计变量")
        self.param_form = QFormLayout(grp_param)
        for f in FEATURE_ORDER:
            meta = FEATURE_META.get(f, {'symbol': f, 'label': f, 'range': '', 'type': 'float', 'min': 0.0, 'max': 999.0})
            lbl = QLabel(f"{meta['symbol']} {meta['label']}:")
            if meta.get('range'):
                lbl.setToolTip(f"取值范围: {meta['range']}")

            row_widget = QWidget(grp_param)
            row_widget.setToolTip(f"取值范围: {meta['range']}")
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)

            if meta.get('type') == 'percent':
                slider = QSlider(Qt.Horizontal, row_widget)
                slider.setRange(int(meta['min']), int(meta['max']))
                slider.setSingleStep(int(meta.get('step', 1)))
                slider.setTickPosition(QSlider.TicksBelow)
                slider.setTickInterval(int(meta.get('step', 1)))

                val_lbl = QLabel(f"{slider.value()}%", row_widget)
                val_lbl.setFixedWidth(50)
                val_lbl.setStyleSheet("color: #2980b9; font-weight: bold;")

                def make_snap(s=slider, l=val_lbl, step=int(meta.get('step', 1))):
                    def snap(v):
                        sn = round(v / step) * step
                        s.blockSignals(True)
                        s.setValue(sn)
                        s.blockSignals(False)
                        l.setText(f"{sn}%")
                    return snap

                slider.valueChanged.connect(make_snap())
                slider.sliderReleased.connect(self.invalidate_prediction)
                row_layout.addWidget(slider)
                row_layout.addWidget(val_lbl)
                self.inputs[f], self.rows[f] = slider, (lbl, row_widget, val_lbl)
            else:
                edit = QLineEdit("0", row_widget)
                edit.setPlaceholderText(meta.get('range', ''))
                if meta.get('type') == 'int':
                    edit.setValidator(QIntValidator(int(meta['min']), int(meta['max']), edit))
                else:
                    validator = QDoubleValidator(float(meta['min']), float(meta['max']), int(meta.get('decimals', 2)), edit)
                    validator.setNotation(QDoubleValidator.StandardNotation)
                    edit.setValidator(validator)
                edit.textEdited.connect(self.invalidate_prediction)
                row_layout.addWidget(edit)
                self.inputs[f], self.rows[f] = edit, (lbl, row_widget, None)

            self.param_form.addRow(lbl, row_widget)
        vbox.addWidget(grp_param)

        grp_display = QGroupBox("3. 环境显示")
        display_layout = QVBoxLayout()
        self.chk_upper = QCheckBox("显示纵向叠加层 (楼上)")
        self.chk_side = QCheckBox("显示横向相邻间 (并排)")
        self.btn_reset_view = QPushButton("重置为默认视角")
        self.btn_reset_view.setToolTip("点击后将 3D 视角恢复到初始默认角度 (elev=25, azim=45)")
        self.btn_reset_view.clicked.connect(self.reset_3d_view)
        self.chk_upper.stateChanged.connect(self.update_viz_only)
        self.chk_side.stateChanged.connect(self.update_viz_only)
        display_layout.addWidget(self.chk_upper)
        display_layout.addWidget(self.chk_side)
        display_layout.addWidget(self.btn_reset_view)
        grp_display.setLayout(display_layout)
        vbox.addWidget(grp_display)

        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("⚡ 执行 Stacking 集成预测")
        self.btn_run.setFixedHeight(50)
        self.btn_run.setStyleSheet("background-color: #2c3e50; color: white; font-weight: bold;")
        self.btn_run.clicked.connect(self.do_prediction)

        self.btn_save_scheme = QPushButton("💾 保存当前方案")
        self.btn_save_scheme.setFixedHeight(50)
        self.btn_save_scheme.setEnabled(False)
        self.btn_save_scheme.setStyleSheet("background-color: #16a085; color: white; font-weight: bold;")
        self.btn_save_scheme.clicked.connect(self.save_current_scheme)

        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_save_scheme)
        vbox.addLayout(btn_row)
        vbox.addStretch()

        scroll.setWidget(left_container)
        main_layout.addWidget(scroll)

        right_box = QVBoxLayout()

        res_grp = QGroupBox("预测看板")
        res_grid = QHBoxLayout()
        self.res_labels = {}
        for key in TARGET_KEYS:
            v_unit = QVBoxLayout()
            title = QLabel(TARGET_DISPLAY[key])
            title.setAlignment(Qt.AlignCenter)
            lbl_val = QLabel("--")
            lbl_val.setAlignment(Qt.AlignCenter)
            lbl_val.setStyleSheet("font-size: 20px; font-weight: bold; color: #d35400;")
            self.res_labels[key] = lbl_val
            v_unit.addWidget(title)
            v_unit.addWidget(lbl_val)
            res_grid.addLayout(v_unit)
        res_grp.setLayout(res_grid)
        right_box.addWidget(res_grp)

        compare_grp = QGroupBox("方案保存与性能对比")
        compare_layout = QVBoxLayout()

        selector_row = QHBoxLayout()
        self.cb_compare_base = QComboBox()
        self.cb_compare_current = QComboBox()
        self.cb_compare_base.currentTextChanged.connect(self.update_comparison_panel)
        self.cb_compare_current.currentTextChanged.connect(self.update_comparison_panel)
        selector_row.addWidget(QLabel("基准方案:"))
        selector_row.addWidget(self.cb_compare_base)
        selector_row.addWidget(QLabel("对比方案:"))
        selector_row.addWidget(self.cb_compare_current)
        compare_layout.addLayout(selector_row)

        self.lbl_compare_hint = QLabel("请先执行预测，并至少保存两个方案后再进行对比。")
        self.lbl_compare_hint.setStyleSheet("color: #7f8c8d;")
        compare_layout.addWidget(self.lbl_compare_hint)

        compare_grid = QGridLayout()
        compare_grid.addWidget(QLabel("指标"), 0, 0)
        compare_grid.addWidget(QLabel("基准方案"), 0, 1)
        compare_grid.addWidget(QLabel("对比方案"), 0, 2)
        compare_grid.addWidget(QLabel("变化值"), 0, 3)

        for row_idx, key in enumerate(TARGET_KEYS, start=1):
            title = QLabel(TARGET_DISPLAY[key])
            base_lbl = QLabel("--")
            curr_lbl = QLabel("--")
            delta_lbl = QLabel("--")
            for lbl in [base_lbl, curr_lbl, delta_lbl]:
                lbl.setAlignment(Qt.AlignCenter)
            delta_lbl.setStyleSheet("font-weight: bold; color: #7f8c8d;")
            self.compare_labels[key] = {
                'base': base_lbl,
                'current': curr_lbl,
                'delta': delta_lbl
            }
            compare_grid.addWidget(title, row_idx, 0)
            compare_grid.addWidget(base_lbl, row_idx, 1)
            compare_grid.addWidget(curr_lbl, row_idx, 2)
            compare_grid.addWidget(delta_lbl, row_idx, 3)

        compare_layout.addLayout(compare_grid)
        compare_grp.setLayout(compare_layout)
        right_box.addWidget(compare_grp)

        self.fig = plt.figure(figsize=(9, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.view_init(elev=25, azim=45)
        self.canvas = FigureCanvas(self.fig)
        right_box.addWidget(self.canvas)
        main_layout.addLayout(right_box)

    def format_metric(self, key, value):
        return f"{float(value):.2f}"

    def clear_comparison_panel(self, hint_text):
        self.lbl_compare_hint.setText(hint_text)
        for labels in self.compare_labels.values():
            labels['base'].setText("--")
            labels['current'].setText("--")
            labels['delta'].setText("--")
            labels['delta'].setStyleSheet("font-weight: bold; color: #7f8c8d;")

    def get_saved_scheme(self, name):
        for scheme in self.saved_schemes:
            if scheme['name'] == name:
                return scheme
        return None

    def make_scheme_name_unique(self, base_name):
        existing_names = {scheme['name'] for scheme in self.saved_schemes}
        if base_name not in existing_names:
            return base_name

        idx = 2
        while f"{base_name} ({idx})" in existing_names:
            idx += 1
        return f"{base_name} ({idx})"

    def refresh_saved_scheme_selectors(self, base_name=None, current_name=None):
        names = [scheme['name'] for scheme in self.saved_schemes]
        for combo in [self.cb_compare_base, self.cb_compare_current]:
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            combo.blockSignals(False)

        if names:
            if len(names) >= 2:
                self.cb_compare_base.setCurrentText(base_name if base_name in names else names[-2])
                self.cb_compare_current.setCurrentText(current_name if current_name in names else names[-1])
            else:
                self.cb_compare_base.setCurrentText(names[0])
                self.cb_compare_current.setCurrentText(names[0])

        self.update_comparison_panel()

    def update_comparison_panel(self, *args):
        if len(self.saved_schemes) < 2:
            self.clear_comparison_panel("请至少保存两个方案后再进行性能对比。")
            return

        base_name = self.cb_compare_base.currentText().strip()
        current_name = self.cb_compare_current.currentText().strip()
        if not base_name or not current_name:
            self.clear_comparison_panel("请选择两个已保存方案。")
            return

        if base_name == current_name:
            self.clear_comparison_panel("请选择两个不同的已保存方案进行对比。")
            return

        base_scheme = self.get_saved_scheme(base_name)
        current_scheme = self.get_saved_scheme(current_name)
        if not base_scheme or not current_scheme:
            self.clear_comparison_panel("方案数据不存在，请重新选择。")
            return

        self.lbl_compare_hint.setText(
            f"变化值 = 对比方案 - 基准方案 | {base_name} vs {current_name}"
        )

        for key in TARGET_KEYS:
            base_val = float(base_scheme['results'][key])
            current_val = float(current_scheme['results'][key])
            delta_val = current_val - base_val

            if abs(delta_val) < 1e-9:
                delta_color = "#7f8c8d"
            elif delta_val > 0:
                delta_color = "#d35400"
            else:
                delta_color = "#2980b9"

            self.compare_labels[key]['base'].setText(self.format_metric(key, base_val))
            self.compare_labels[key]['current'].setText(self.format_metric(key, current_val))
            self.compare_labels[key]['delta'].setText(f"{delta_val:+.2f}")
            self.compare_labels[key]['delta'].setStyleSheet(
                f"font-weight: bold; color: {delta_color};"
            )

    def save_current_scheme(self):
        if not self.last_prediction:
            QMessageBox.information(self, "暂无可保存方案", "请先执行一次预测，再保存当前设计方案。")
            return

        default_name = (
            f"方案{len(self.saved_schemes) + 1} - "
            f"{self.last_prediction['ori'].split()[0]} / {self.last_prediction['shade'].split()[0]}"
        )
        scheme_name, ok = QInputDialog.getText(
            self,
            "保存设计方案",
            "请输入方案名称：",
            text=default_name
        )
        if not ok:
            return

        scheme_name = self.make_scheme_name_unique((scheme_name or default_name).strip() or default_name)
        previous_name = self.saved_schemes[-1]['name'] if self.saved_schemes else None

        self.saved_schemes.append({
            'name': scheme_name,
            'ori': self.last_prediction['ori'],
            'shade': self.last_prediction['shade'],
            'raw_inputs': dict(self.last_prediction['raw_inputs']),
            'calculated': dict(self.last_prediction['calculated']),
            'results': dict(self.last_prediction['results'])
        })

        base_name = previous_name if previous_name else scheme_name
        self.refresh_saved_scheme_selectors(base_name=base_name, current_name=scheme_name)

        if len(self.saved_schemes) == 1:
            msg = f"{scheme_name} 已保存。继续保存新的方案后，就可以在右侧直接做性能对比。"
        else:
            msg = f"{scheme_name} 已保存，并已与上一版方案建立对比。"
        QMessageBox.information(self, "方案已保存", msg)

    def apply_presets(self):
        ori, sha = self.cb_ori.currentText(), self.cb_sha.currentText()
        preset = PRESET_DATA.get((ori, sha), {})

        for feat, val in preset.items():
            if feat in self.inputs:
                w = self.inputs[feat]
                if isinstance(w, QSlider):
                    w.blockSignals(True)
                    w.setValue(int(float(val)))
                    _, _, sub_lbl = self.rows[feat]
                    if sub_lbl:
                        sub_lbl.setText(f"{int(float(val))}%")
                    w.blockSignals(False)
                else:
                    w.setText(str(val))

        active_list = SHADE_FEATURE_MAP.get(sha, FEATURE_ORDER)
        is_p = "垂直" in sha or "组合" in sha

        for f, (lbl, row_widget, sub_lbl) in self.rows.items():
            vis = f in active_list or (f == 'WWR_s_exp' and is_p)
            lbl.setVisible(vis)
            row_widget.setVisible(vis)

            if f == 'WWR_s':
                main_ctrl = self.inputs[f]
                if isinstance(main_ctrl, QSlider):
                    main_ctrl.setEnabled(not is_p)
                else:
                    main_ctrl.setReadOnly(is_p)

    def reset_3d_view(self):
        self.ax.view_init(elev=25, azim=45)
        self.canvas.draw()

    def update_viz_only(self):
        sha_mode = self.cb_sha.currentText()
        raw_in = {f: self.get_feature_value(f) for f in FEATURE_ORDER}
        cal_data = geometry_engine(raw_in, sha_mode)
        saved_elev, saved_azim = self.ax.elev, self.ax.azim
        draw_classroom_3d(
            self.ax,
            cal_data,
            sha_mode,
            self.chk_upper.isChecked(),
            self.chk_side.isChecked()
        )
        self.ax.view_init(elev=saved_elev, azim=saved_azim)
        self.canvas.draw()

    def do_prediction(self):
        ori_mode = self.cb_ori.currentText()
        sha_mode = self.cb_sha.currentText()
        folder = self.model_map.get((ori_mode, sha_mode))
        if not folder:
            QMessageBox.warning(self, "模型缺失", "当前朝向与遮阳形式尚未找到对应模型目录。")
            return

        self.last_prediction = None
        self.btn_save_scheme.setEnabled(False)

        try:
            self.btn_run.setEnabled(False)
            QApplication.processEvents()

            raw_in = {f: self.get_feature_value(f) for f in FEATURE_ORDER}
            cal_data = geometry_engine(raw_in, sha_mode)

            if "悬挑" in sha_mode or "组合" in sha_mode:
                alpha_val = self.get_feature_value('alpha_oh')
                if alpha_val < 0:
                    shs_val = self.get_feature_value('SH_s')
                    dmv_val = self.get_feature_value('d_mv')
                    max_L_oh = round(dmv_val + shs_val, 2)
                    input_L_oh = self.get_feature_value('L_oh')
                    if input_L_oh > max_L_oh:
                        QMessageBox.warning(
                            self,
                            "遮阳板长度约束",
                            f"当 alpha_oh < 0 时，L_oh 必须 ≤ d_mv + SH_s\n\n"
                            f"  当前 d_mv = {dmv_val},  SH_s = {shs_val}\n"
                            f"  最大允许值 = d_mv + SH_s = {max_L_oh}\n"
                            f"  您输入的 L_oh = {input_L_oh}\n\n"
                            f"L_oh 已自动限制为 {cal_data['L_oh']}"
                        )

            if "垂直" in sha_mode or "组合" in sha_mode:
                input_mtl = raw_in.get('M_TL', 0.0)
                input_mtr = raw_in.get('M_TR', 0.0)
                wv_val = raw_in.get('W_v', 0.01)
                if input_mtl + input_mtr > wv_val:
                    QMessageBox.warning(
                        self,
                        "垂直遮阳移动约束",
                        f"M_TL + M_TR 之和不能超过 W_v\n\n"
                        f"  当前 W_v = {wv_val}\n"
                        f"  您输入的 M_TL = {input_mtl},  M_TR = {input_mtr}\n"
                        f"  合计 = {round(input_mtl + input_mtr, 2)}  >  W_v = {wv_val}\n\n"
                        f"已按比例自动缩放：M_TL = {cal_data['M_TL']},  M_TR = {cal_data['M_TR']}"
                    )

            w_wwrc = self.inputs['WWR_c']
            if isinstance(w_wwrc, QSlider):
                w_wwrc.blockSignals(True)
                w_wwrc.setValue(cal_data['WWR_c'])
                _, _, lbl_c = self.rows['WWR_c']
                if lbl_c:
                    lbl_c.setText(f"{cal_data['WWR_c']}%")
                w_wwrc.blockSignals(False)

            w_wwrs = self.inputs['WWR_s']
            if isinstance(w_wwrs, QSlider):
                w_wwrs.blockSignals(True)
                w_wwrs.setValue(cal_data['WWR_s'])
                _, _, slbl_s = self.rows['WWR_s']
                if slbl_s:
                    slbl_s.setText(f"{cal_data['WWR_s']}%")
                w_wwrs.blockSignals(False)
            else:
                w_wwrs.setText(str(cal_data['WWR_s']))

            if 'H_s' in self.inputs:
                self.inputs['H_s'].setText(str(cal_data['H_s']))
            if 'W_c' in self.inputs:
                self.inputs['W_c'].setText(str(cal_data['W_c']))
            if 'W_s' in self.inputs:
                self.inputs['W_s'].setText(str(cal_data['W_s']))
            if 'SH_s' in self.inputs:
                self.inputs['SH_s'].setText(str(cal_data['SH_s']))
            if 'SH_c' in self.inputs:
                self.inputs['SH_c'].setText(str(cal_data['SH_c']))
            if 'd_A_s' in self.inputs:
                self.inputs['d_A_s'].setText(str(cal_data['d_A_s']))
            if 'd_B_s' in self.inputs:
                self.inputs['d_B_s'].setText(str(cal_data['d_B_s']))
            if 'L_oh' in self.inputs:
                self.inputs['L_oh'].setText(str(cal_data['L_oh']))
            if 'd_mv' in self.inputs:
                self.inputs['d_mv'].setText(str(cal_data['d_mv']))
            if 'M_TL' in self.inputs:
                self.inputs['M_TL'].setText(str(cal_data['M_TL']))
            if 'M_TR' in self.inputs:
                self.inputs['M_TR'].setText(str(cal_data['M_TR']))

            active_feats = SHADE_FEATURE_MAP.get(sha_mode, FEATURE_ORDER[1:13])
            X = pd.DataFrame([[cal_data[f] for f in active_feats]], columns=active_feats)

            ckey = (ori_mode, sha_mode)
            if ckey not in self.model_cache:
                self.model_cache[ckey] = {
                    f"{t}_{m}": joblib.load(os.path.join(folder, f"{m}_model_{t}.joblib"))
                    for t in TARGET_KEYS for m in ["xgb", "lgbm", "rf", "meta"]
                }

            mc = self.model_cache[ckey]
            result_values = {}
            for t in TARGET_KEYS:
                m1 = mc[f"{t}_xgb"].predict(X)
                m2 = mc[f"{t}_lgbm"].predict(X)
                m3 = mc[f"{t}_rf"].predict(X)
                final = mc[f"{t}_meta"].predict(np.column_stack([m1, m2, m3]))[0]

                if t in ["sDA", "sGA"]:
                    final = max(0.0, min(final, 1.0))
                    v_show = final * 100
                elif t in ["UDI", "SVF"]:
                    v_show = max(0.0, min(final, 100.0))
                else:
                    v_show = final

                result_values[t] = float(v_show)
                self.res_labels[t].setText(f"{v_show:.2f}")

            saved_elev, saved_azim = self.ax.elev, self.ax.azim
            draw_classroom_3d(
                self.ax,
                cal_data,
                sha_mode,
                self.chk_upper.isChecked(),
                self.chk_side.isChecked()
            )
            self.ax.view_init(elev=saved_elev, azim=saved_azim)
            self.canvas.draw()

            self.last_prediction = {
                'ori': ori_mode,
                'shade': sha_mode,
                'raw_inputs': dict(raw_in),
                'calculated': dict(cal_data),
                'results': result_values
            }
            self.btn_save_scheme.setEnabled(True)
            self.btn_run.setEnabled(True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "运行故障", f"预测或计算出错：\n{str(e)}")
            self.btn_save_scheme.setEnabled(False)
            self.btn_run.setEnabled(True)


# %% [Cell 4: GUI 界面与预测逻辑]
class ClassroomPredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EduSync-SZ：深圳中小学校教室能耗与采光性能智能预测系统")
        self.resize(1500, 950)
        
        self.model_map = {}
        self.model_cache = {}
        self.inputs = {}
        self.rows = {} 
        
        self.scan_folders()
        self.init_ui()
        self.apply_presets()

    def scan_folders(self):
        if not os.path.exists(BASE_PATH): return
        folders = [d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d))]
        for f in folders:
            ori = ORI_LIST[0] if "South0°" in f else (ORI_LIST[1] if "North0°" in f else None)
            if not ori: continue
            sha = SHADE_LIST[0]
            if "Overhang+Vertical" in f: sha = SHADE_LIST[3]
            elif "Overhang" in f: sha = SHADE_LIST[1]
            elif "Vertical" in f: sha = SHADE_LIST[2]
            elif "Frame" in f: sha = SHADE_LIST[4]
            self.model_map[(ori, sha)] = os.path.join(BASE_PATH, f)

        # 北向组合遮阳固定使用替换后的模型目录
        preferred_folder = "0312_2000_North0°_Overhang+Vertical(1)_replaced"
        preferred_path = os.path.join(BASE_PATH, preferred_folder)
        if os.path.isdir(preferred_path):
            self.model_map[(ORI_LIST[1], SHADE_LIST[3])] = preferred_path

    def get_feature_value(self, feat):
        w = self.inputs[feat]
        meta = FEATURE_META.get(feat, {})
        min_v = meta.get('min')
        max_v = meta.get('max')

        if isinstance(w, QSlider):
            return float(w.value())

        raw_text = w.text().strip()
        if not raw_text:
            raw_val = float(min_v) if min_v is not None else 0.0
        else:
            try:
                raw_val = float(raw_text)
            except ValueError:
                raw_val = float(min_v) if min_v is not None else 0.0

        if min_v is not None:
            raw_val = max(float(min_v), raw_val)
        if max_v is not None:
            raw_val = min(float(max_v), raw_val)

        if meta.get('type') == 'int':
            raw_val = int(round(raw_val))
        else:
            decimals = int(meta.get('decimals', 2))
            raw_val = round(raw_val, decimals)

        return float(raw_val)

    def init_ui(self):
        main_widget = QWidget(); self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        scroll = QScrollArea(); scroll.setFixedWidth(400); scroll.setWidgetResizable(True)
        left_container = QWidget(); vbox = QVBoxLayout(left_container)
        
        # 1. 策略选择
        grp_sel = QGroupBox("1. 策略选择"); f_sel = QFormLayout(grp_sel)
        self.cb_ori, self.cb_sha = QComboBox(), QComboBox()
        self.cb_ori.addItems(ORI_LIST); self.cb_sha.addItems(SHADE_LIST)
        self.cb_ori.currentTextChanged.connect(self.apply_presets)
        self.cb_sha.currentTextChanged.connect(self.apply_presets)
        f_sel.addRow("朝向:", self.cb_ori); f_sel.addRow("形式:", self.cb_sha)
        vbox.addWidget(grp_sel)

        # 2. 设计变量
        grp_param = QGroupBox("2. 设计变量"); self.param_form = QFormLayout(grp_param)
        for f in FEATURE_ORDER:
            meta = FEATURE_META.get(f, {'symbol': f, 'label': f, 'range': '', 'type': 'float', 'min': 0.0, 'max': 999.0})
            lbl = QLabel(f"{meta['symbol']} {meta['label']}:")
            if meta.get('range'):
                lbl.setToolTip(f"取值范围: {meta['range']}")
            row_widget = QWidget(grp_param)
            row_widget.setToolTip(f"取值范围: {meta['range']}")
            row_layout = QHBoxLayout(row_widget); row_layout.setContentsMargins(0, 0, 0, 0)
            
            if meta.get('type') == 'percent':
                slider = QSlider(Qt.Horizontal, row_widget)
                slider.setRange(int(meta['min']), int(meta['max']))
                slider.setSingleStep(int(meta.get('step', 1)))
                slider.setTickPosition(QSlider.TicksBelow)
                slider.setTickInterval(int(meta.get('step', 1)))
                
                val_lbl = QLabel(f"{slider.value()}%", row_widget)
                val_lbl.setFixedWidth(50); val_lbl.setStyleSheet("color: #2980b9; font-weight: bold;")
                
                def make_snap(s=slider, l=val_lbl, step=int(meta.get('step', 1))):
                    def snap(v):
                        sn = round(v / step) * step
                        s.blockSignals(True); s.setValue(sn); s.blockSignals(False)
                        l.setText(f"{sn}%")
                    return snap
                
                slider.valueChanged.connect(make_snap())
                row_layout.addWidget(slider); row_layout.addWidget(val_lbl)
                self.inputs[f], self.rows[f] = slider, (lbl, row_widget, val_lbl)
            else:
                edit = QLineEdit("0", row_widget)
                edit.setPlaceholderText(meta.get('range', ''))
                if meta.get('type') == 'int':
                    edit.setValidator(QIntValidator(int(meta['min']), int(meta['max']), edit))
                else:
                    validator = QDoubleValidator(float(meta['min']), float(meta['max']), int(meta.get('decimals', 2)), edit)
                    validator.setNotation(QDoubleValidator.StandardNotation)
                    edit.setValidator(validator)
                row_layout.addWidget(edit)
                self.inputs[f], self.rows[f] = edit, (lbl, row_widget, None)
            
            self.param_form.addRow(lbl, row_widget)
        vbox.addWidget(grp_param)
        
        # 3. 环境显示控制 (打勾后自动重绘 3D，不需要跑模型)
        grp_display = QGroupBox("3. 环境显示"); display_layout = QVBoxLayout()
        self.chk_upper = QCheckBox("显示纵向叠加层 (楼上)")
        self.chk_side = QCheckBox("显示横向相邻间 (并排)")
        self.btn_reset_view = QPushButton("重置为默认视角")
        self.btn_reset_view.setToolTip("点击后将 3D 视角恢复到初始默认角度 (elev=25, azim=45)")
        self.btn_reset_view.clicked.connect(self.reset_3d_view)
        self.chk_upper.stateChanged.connect(self.update_viz_only)
        self.chk_side.stateChanged.connect(self.update_viz_only)
        display_layout.addWidget(self.chk_upper); display_layout.addWidget(self.chk_side)
        display_layout.addWidget(self.btn_reset_view)
        grp_display.setLayout(display_layout); vbox.addWidget(grp_display)

        # 4. 预测按钮
        self.btn_run = QPushButton("⚡ 执行 Stacking 集成预测")
        self.btn_run.setFixedHeight(50)
        self.btn_run.setStyleSheet("background-color: #2c3e50; color: white; font-weight: bold;")
        self.btn_run.clicked.connect(self.do_prediction)
        vbox.addWidget(self.btn_run); vbox.addStretch()

        scroll.setWidget(left_container); main_layout.addWidget(scroll)

        # 右侧面板
        right_box = QVBoxLayout()
        res_grp = QGroupBox("预测看板"); res_grid = QHBoxLayout()
        self.res_labels = {}
        for key in TARGET_KEYS:
            v_unit = QVBoxLayout()
            title = QLabel(TARGET_DISPLAY[key]); title.setAlignment(Qt.AlignCenter)
            lbl_val = QLabel("--"); lbl_val.setAlignment(Qt.AlignCenter); lbl_val.setStyleSheet("font-size: 20px; font-weight: bold; color: #d35400;")
            self.res_labels[key] = lbl_val
            v_unit.addWidget(title); v_unit.addWidget(lbl_val); res_grid.addLayout(v_unit)
        res_grp.setLayout(res_grid); right_box.addWidget(res_grp)

        self.fig = plt.figure(figsize=(9, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.view_init(elev=25, azim=45)
        self.canvas = FigureCanvas(self.fig); right_box.addWidget(self.canvas)
        main_layout.addLayout(right_box)

    def apply_presets(self):
        ori, sha = self.cb_ori.currentText(), self.cb_sha.currentText()
        preset = PRESET_DATA.get((ori, sha), {})
        
        for feat, val in preset.items():
            if feat in self.inputs:
                w = self.inputs[feat]
                if isinstance(w, QSlider):
                    w.blockSignals(True); w.setValue(int(float(val)))
                    _, _, sub_lbl = self.rows[feat]
                    if sub_lbl: sub_lbl.setText(f"{int(float(val))}%")
                    w.blockSignals(False)
                else: w.setText(str(val))
        
        active_list = SHADE_FEATURE_MAP.get(sha, FEATURE_ORDER)
        is_p = "垂直" in sha or "组合" in sha

        for f, (lbl, row_widget, sub_lbl) in self.rows.items():
            vis = f in active_list or (f == 'WWR_s_exp' and is_p)
            lbl.setVisible(vis); row_widget.setVisible(vis)
            
            if f == 'WWR_s':
                main_ctrl = self.inputs[f]
                if isinstance(main_ctrl, QSlider): main_ctrl.setEnabled(not is_p)
                else: main_ctrl.setReadOnly(is_p)

    def reset_3d_view(self):
        self.ax.view_init(elev=25, azim=45)
        self.canvas.draw()

    def update_viz_only(self):
        sha_mode = self.cb_sha.currentText()
        raw_in = {f: self.get_feature_value(f) for f in FEATURE_ORDER}
        cal_data = geometry_engine(raw_in, sha_mode)
        saved_elev, saved_azim = self.ax.elev, self.ax.azim
        draw_classroom_3d(self.ax, cal_data, sha_mode, self.chk_upper.isChecked(), self.chk_side.isChecked())
        self.ax.view_init(elev=saved_elev, azim=saved_azim)
        self.canvas.draw()

    def do_prediction(self):
        sha_mode = self.cb_sha.currentText()
        folder = self.model_map.get((self.cb_ori.currentText(), sha_mode))
        if not folder: return
        
        try:
            self.btn_run.setEnabled(False); QApplication.processEvents()
            
            raw_in = {f: self.get_feature_value(f) for f in FEATURE_ORDER}
            cal_data = geometry_engine(raw_in, sha_mode)
            
            # --- L_oh 约束检查与用户提示 ---
            if ("悬挑" in sha_mode or "组合" in sha_mode):
                alpha_val = self.get_feature_value('alpha_oh')
                if alpha_val < 0:
                    shs_val = self.get_feature_value('SH_s')
                    dmv_val = self.get_feature_value('d_mv')
                    max_L_oh = round(dmv_val + shs_val, 2)
                    input_L_oh = self.get_feature_value('L_oh')
                    if input_L_oh > max_L_oh:
                        QMessageBox.warning(self, "遮阳板长度约束",
                            f"当 alpha_oh < 0 时，L_oh 必须 ≤ d_mv + SH_s\n\n"
                            f"  当前 d_mv = {dmv_val},  SH_s = {shs_val}\n"
                            f"  最大允许值 = d_mv + SH_s = {max_L_oh}\n"
                            f"  您输入的 L_oh = {input_L_oh}\n\n"
                            f"L_oh 已自动限制为 {cal_data['L_oh']}")

            if "垂直" in sha_mode or "组合" in sha_mode:
                input_mtl = raw_in.get('M_TL', 0.0)
                input_mtr = raw_in.get('M_TR', 0.0)
                wv_val = raw_in.get('W_v', 0.01)
                if input_mtl + input_mtr > wv_val:
                    QMessageBox.warning(self, "垂直遮阳移动约束",
                        f"M_TL + M_TR 之和不能超过 W_v\n\n"
                        f"  当前 W_v = {wv_val}\n"
                        f"  您输入的 M_TL = {input_mtl},  M_TR = {input_mtr}\n"
                        f"  合计 = {round(input_mtl + input_mtr, 2)}  >  W_v = {wv_val}\n\n"
                        f"已按比例自动缩放：M_TL = {cal_data['M_TL']},  M_TR = {cal_data['M_TR']}")

            # --- 回写界面数据 ---
            w_wwrc = self.inputs['WWR_c']
            if isinstance(w_wwrc, QSlider):
                w_wwrc.blockSignals(True); w_wwrc.setValue(cal_data['WWR_c'])
                _, _, lbl_c = self.rows['WWR_c']
                if lbl_c: lbl_c.setText(f"{cal_data['WWR_c']}%")
                w_wwrc.blockSignals(False)
            
            w_wwrs = self.inputs['WWR_s']
            if isinstance(w_wwrs, QSlider):
                w_wwrs.blockSignals(True); w_wwrs.setValue(cal_data['WWR_s'])
                _, _, slbl_s = self.rows['WWR_s']
                if slbl_s: slbl_s.setText(f"{cal_data['WWR_s']}%")
                w_wwrs.blockSignals(False)
            else: w_wwrs.setText(str(cal_data['WWR_s']))
            
            # 将物理限制自动回写
            if 'H_s' in self.inputs: self.inputs['H_s'].setText(str(cal_data['H_s']))
            if 'W_c' in self.inputs: self.inputs['W_c'].setText(str(cal_data['W_c']))
            if 'W_s' in self.inputs: self.inputs['W_s'].setText(str(cal_data['W_s']))
            if 'SH_s' in self.inputs: self.inputs['SH_s'].setText(str(cal_data['SH_s']))
            if 'SH_c' in self.inputs: self.inputs['SH_c'].setText(str(cal_data['SH_c']))
            if 'd_A_s' in self.inputs: self.inputs['d_A_s'].setText(str(cal_data['d_A_s']))
            if 'd_B_s' in self.inputs: self.inputs['d_B_s'].setText(str(cal_data['d_B_s']))
            if 'L_oh' in self.inputs: self.inputs['L_oh'].setText(str(cal_data['L_oh']))
            if 'd_mv' in self.inputs: self.inputs['d_mv'].setText(str(cal_data['d_mv']))
            if 'M_TL' in self.inputs: self.inputs['M_TL'].setText(str(cal_data['M_TL']))
            if 'M_TR' in self.inputs: self.inputs['M_TR'].setText(str(cal_data['M_TR']))
            
            # --- 执行 Stacking 预测 ---
            active_feats = SHADE_FEATURE_MAP.get(sha_mode, FEATURE_ORDER[1:13])
            X = pd.DataFrame([[cal_data[f] for f in active_feats]], columns=active_feats)
            
            ckey = (self.cb_ori.currentText(), sha_mode)
            if ckey not in self.model_cache:
                self.model_cache[ckey] = {f"{t}_{m}": joblib.load(os.path.join(folder, f"{m}_model_{t}.joblib")) 
                                          for t in TARGET_KEYS for m in ["xgb", "lgbm", "rf", "meta"]}
            
            mc = self.model_cache[ckey]
            for t in TARGET_KEYS:
                m1, m2, m3 = mc[f"{t}_xgb"].predict(X), mc[f"{t}_lgbm"].predict(X), mc[f"{t}_rf"].predict(X)
                final = mc[f"{t}_meta"].predict(np.column_stack([m1, m2, m3]))[0]
                
                if t in ["sDA", "sGA"]:
                    # 训练数据为 0~1 小数，需 clamp 后 ×100 显示
                    final = max(0.0, min(final, 1.0))
                    v_show = final * 100
                elif t in ["UDI", "SVF"]:
                    # 训练数据已为 0~100 百分比，直接 clamp 显示
                    v_show = max(0.0, min(final, 100.0))
                else:
                    v_show = final
                
                self.res_labels[t].setText(f"{v_show:.2f}")

            # --- 更新 3D ---
            saved_elev, saved_azim = self.ax.elev, self.ax.azim
            draw_classroom_3d(self.ax, cal_data, sha_mode, self.chk_upper.isChecked(), self.chk_side.isChecked())
            self.ax.view_init(elev=saved_elev, azim=saved_azim)
            self.canvas.draw()
            
            self.btn_run.setEnabled(True)
        except Exception as e:
            import traceback; traceback.print_exc()
            QMessageBox.critical(self, "运行故障", f"预测或计算出错：\n{str(e)}")
            self.btn_run.setEnabled(True)


# %% [Cell 4C: GUI 增强覆盖 - 全滑杆版]

def _slider_int(real_val, step):
    """实际浮点值 → 滑杆整数刻度"""
    return int(round(float(real_val) / float(step)))

def _slider_real(slider_int, step):
    """滑杆整数刻度 → 实际浮点值"""
    return round(slider_int * float(step), 6)

def _format_val(real_val, step, unit):
    """根据 step 精度格式化显示值"""
    s = str(step)
    if '.' in s:
        decimals = len(s.rstrip('0').split('.')[1])
    else:
        decimals = 0
    return f"{real_val:.{decimals}f} {unit}"


class ClassroomPredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EduSync-SZ：深圳中小学校教室能耗与采光性能智能预测系统")
        self.resize(1500, 950)

        self.model_map = {}
        self.model_cache = {}
        self.inputs = {}
        self.rows = {}
        self.saved_schemes = []
        self.last_prediction = None
        self.compare_labels = {}

        self.scan_folders()
        self.init_ui()
        self.apply_presets()

    def scan_folders(self):
        if not os.path.exists(BASE_PATH):
            return
        folders = [d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d))]
        for f in folders:
            ori = ORI_LIST[0] if "South0°" in f else (ORI_LIST[1] if "North0°" in f else None)
            if not ori:
                continue
            sha = SHADE_LIST[0]
            if "Overhang+Vertical" in f:
                sha = SHADE_LIST[3]
            elif "Overhang" in f:
                sha = SHADE_LIST[1]
            elif "Vertical" in f:
                sha = SHADE_LIST[2]
            elif "Frame" in f:
                sha = SHADE_LIST[4]
            self.model_map[(ori, sha)] = os.path.join(BASE_PATH, f)

        preferred_folder = "0312_2000_North0°_Overhang+Vertical(1)_replaced"
        preferred_path = os.path.join(BASE_PATH, preferred_folder)
        if os.path.isdir(preferred_path):
            self.model_map[(ORI_LIST[1], SHADE_LIST[3])] = preferred_path

    def get_feature_value(self, feat):
        """从滑杆读取实际浮点值"""
        slider = self.inputs[feat]
        meta = FEATURE_META[feat]
        return _slider_real(slider.value(), meta['step'])

    def set_feature_value(self, feat, real_val):
        """将实际浮点值写回滑杆和标签"""
        slider = self.inputs[feat]
        meta = FEATURE_META[feat]
        step = meta['step']
        iv = _slider_int(real_val, step)
        iv = max(slider.minimum(), min(slider.maximum(), iv))
        slider.blockSignals(True)
        slider.setValue(iv)
        slider.blockSignals(False)
        _, _, val_lbl = self.rows[feat]
        if val_lbl:
            val_lbl.setText(_format_val(_slider_real(iv, step), step, meta['unit']))

    def invalidate_prediction(self, *args):
        self.last_prediction = None
        if hasattr(self, 'btn_save_scheme'):
            self.btn_save_scheme.setEnabled(False)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        scroll = QScrollArea()
        scroll.setFixedWidth(460)
        scroll.setWidgetResizable(True)
        left_container = QWidget()
        vbox = QVBoxLayout(left_container)

        grp_sel = QGroupBox("1. 策略选择")
        f_sel = QFormLayout(grp_sel)
        self.cb_ori, self.cb_sha = QComboBox(), QComboBox()
        self.cb_ori.addItems(ORI_LIST)
        self.cb_sha.addItems(SHADE_LIST)
        self.cb_ori.currentTextChanged.connect(self.apply_presets)
        self.cb_sha.currentTextChanged.connect(self.apply_presets)
        self.cb_ori.currentTextChanged.connect(self.invalidate_prediction)
        self.cb_sha.currentTextChanged.connect(self.invalidate_prediction)
        f_sel.addRow("朝向:", self.cb_ori)
        f_sel.addRow("形式:", self.cb_sha)
        vbox.addWidget(grp_sel)

        grp_export = QGroupBox("导出三维模型 (OBJ / 3DS)")
        export_outer = QVBoxLayout(grp_export)
        export_row = QHBoxLayout()
        self.btn_export = QPushButton("导出 3D 模型")
        self.btn_export.setFixedHeight(42)
        self.btn_export.setToolTip("将当前几何导出为 OBJ+MTL 或 3DS；图层与图例一致。本区域在「策略选择」下方，无需滚到底部。")
        self.btn_export.setStyleSheet("background-color: #e67e22; color: white; font-weight: bold;")
        self.btn_export.clicked.connect(self.export_3d_model)
        self.cb_export_fmt = QComboBox()
        self.cb_export_fmt.addItems(["OBJ + MTL", "3DS (3D Studio)"])
        self.cb_export_fmt.setFixedHeight(42)
        self.cb_export_fmt.setFixedWidth(160)
        export_row.addWidget(self.btn_export)
        export_row.addWidget(self.cb_export_fmt)
        export_outer.addLayout(export_row)
        vbox.addWidget(grp_export)

        grp_param = QGroupBox("2. 设计变量")
        self.param_form = QFormLayout(grp_param)

        for feat in FEATURE_ORDER:
            meta = FEATURE_META[feat]
            step = meta['step']
            unit = meta['unit']
            lo = meta['min']
            hi = meta['max']

            lbl = QLabel(f"{meta['symbol']}  {meta['label']}:")
            lbl.setToolTip(f"{lo} ~ {hi} {unit}  (步长 {step})")

            row_widget = QWidget(grp_param)
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)

            slider = QSlider(Qt.Horizontal, row_widget)
            i_lo = _slider_int(lo, step)
            i_hi = _slider_int(hi, step)
            slider.setRange(i_lo, i_hi)
            slider.setSingleStep(1)
            slider.setTickPosition(QSlider.TicksBelow)
            tick_count = i_hi - i_lo
            tick_interval = max(1, tick_count // 10)
            slider.setTickInterval(tick_interval)

            init_real = lo
            val_lbl = QLabel(_format_val(init_real, step, unit), row_widget)
            val_lbl.setFixedWidth(80)
            val_lbl.setStyleSheet("color: #2980b9; font-weight: bold;")

            def make_update(s=slider, l=val_lbl, st=step, u=unit):
                def on_change(v):
                    l.setText(_format_val(_slider_real(v, st), st, u))
                return on_change

            slider.valueChanged.connect(make_update())
            slider.valueChanged.connect(self.invalidate_prediction)

            row_layout.addWidget(slider, 1)
            row_layout.addWidget(val_lbl)
            self.inputs[feat] = slider
            self.rows[feat] = (lbl, row_widget, val_lbl)
            self.param_form.addRow(lbl, row_widget)

        vbox.addWidget(grp_param)

        grp_display = QGroupBox("3. 环境显示")
        display_layout = QVBoxLayout()
        self.chk_upper = QCheckBox("显示纵向叠加层 (楼上)")
        self.chk_side = QCheckBox("显示横向相邻间 (并排)")
        self.btn_reset_view = QPushButton("重置为默认视角")
        self.btn_reset_view.setToolTip("点击后将 3D 视角恢复到初始默认角度 (elev=25, azim=45)")
        self.btn_reset_view.clicked.connect(self.reset_3d_view)
        self.chk_upper.stateChanged.connect(self.update_viz_only)
        self.chk_side.stateChanged.connect(self.update_viz_only)
        display_layout.addWidget(self.chk_upper)
        display_layout.addWidget(self.chk_side)
        display_layout.addWidget(self.btn_reset_view)
        grp_display.setLayout(display_layout)
        vbox.addWidget(grp_display)

        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("⚡ 执行 Stacking 集成预测")
        self.btn_run.setFixedHeight(50)
        self.btn_run.setStyleSheet("background-color: #2c3e50; color: white; font-weight: bold;")
        self.btn_run.clicked.connect(self.do_prediction)

        self.btn_save_scheme = QPushButton("💾 保存当前方案")
        self.btn_save_scheme.setFixedHeight(50)
        self.btn_save_scheme.setEnabled(False)
        self.btn_save_scheme.setStyleSheet("background-color: #16a085; color: white; font-weight: bold;")
        self.btn_save_scheme.clicked.connect(self.save_current_scheme)

        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_save_scheme)
        vbox.addLayout(btn_row)
        vbox.addStretch()

        scroll.setWidget(left_container)
        main_layout.addWidget(scroll)

        right_box = QVBoxLayout()

        res_grp = QGroupBox("预测看板")
        res_grid = QHBoxLayout()
        self.res_labels = {}
        for key in TARGET_KEYS:
            v_unit = QVBoxLayout()
            title = QLabel(TARGET_DISPLAY[key])
            title.setAlignment(Qt.AlignCenter)
            lbl_val = QLabel("--")
            lbl_val.setAlignment(Qt.AlignCenter)
            lbl_val.setStyleSheet("font-size: 20px; font-weight: bold; color: #d35400;")
            self.res_labels[key] = lbl_val
            v_unit.addWidget(title)
            v_unit.addWidget(lbl_val)
            res_grid.addLayout(v_unit)
        res_grp.setLayout(res_grid)
        right_box.addWidget(res_grp)

        compare_grp = QGroupBox("方案保存与性能对比")
        compare_layout = QVBoxLayout()

        selector_row = QHBoxLayout()
        self.cb_compare_base = QComboBox()
        self.cb_compare_current = QComboBox()
        self.cb_compare_base.currentTextChanged.connect(self.update_comparison_panel)
        self.cb_compare_current.currentTextChanged.connect(self.update_comparison_panel)
        selector_row.addWidget(QLabel("基准方案:"))
        selector_row.addWidget(self.cb_compare_base)
        selector_row.addWidget(QLabel("对比方案:"))
        selector_row.addWidget(self.cb_compare_current)
        compare_layout.addLayout(selector_row)

        self.lbl_compare_hint = QLabel("请先执行预测，并至少保存两个方案后再进行对比。")
        self.lbl_compare_hint.setStyleSheet("color: #7f8c8d;")
        compare_layout.addWidget(self.lbl_compare_hint)

        compare_grid = QGridLayout()
        compare_grid.addWidget(QLabel("指标"), 0, 0)
        compare_grid.addWidget(QLabel("基准方案"), 0, 1)
        compare_grid.addWidget(QLabel("对比方案"), 0, 2)
        compare_grid.addWidget(QLabel("变化值"), 0, 3)

        for row_idx, key in enumerate(TARGET_KEYS, start=1):
            title = QLabel(TARGET_DISPLAY[key])
            base_lbl = QLabel("--")
            curr_lbl = QLabel("--")
            delta_lbl = QLabel("--")
            for l in [base_lbl, curr_lbl, delta_lbl]:
                l.setAlignment(Qt.AlignCenter)
            delta_lbl.setStyleSheet("font-weight: bold; color: #7f8c8d;")
            self.compare_labels[key] = {'base': base_lbl, 'current': curr_lbl, 'delta': delta_lbl}
            compare_grid.addWidget(title, row_idx, 0)
            compare_grid.addWidget(base_lbl, row_idx, 1)
            compare_grid.addWidget(curr_lbl, row_idx, 2)
            compare_grid.addWidget(delta_lbl, row_idx, 3)

        compare_layout.addLayout(compare_grid)
        compare_grp.setLayout(compare_layout)
        right_box.addWidget(compare_grp)

        self.fig = plt.figure(figsize=(9, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.view_init(elev=25, azim=45)
        self.canvas = FigureCanvas(self.fig)
        right_box.addWidget(self.canvas)
        main_layout.addLayout(right_box)

    def format_metric(self, key, value):
        return f"{float(value):.2f}"

    def clear_comparison_panel(self, hint_text):
        self.lbl_compare_hint.setText(hint_text)
        for labels in self.compare_labels.values():
            labels['base'].setText("--")
            labels['current'].setText("--")
            labels['delta'].setText("--")
            labels['delta'].setStyleSheet("font-weight: bold; color: #7f8c8d;")

    def get_saved_scheme(self, name):
        for scheme in self.saved_schemes:
            if scheme['name'] == name:
                return scheme
        return None

    def make_scheme_name_unique(self, base_name):
        existing_names = {scheme['name'] for scheme in self.saved_schemes}
        if base_name not in existing_names:
            return base_name
        idx = 2
        while f"{base_name} ({idx})" in existing_names:
            idx += 1
        return f"{base_name} ({idx})"

    def refresh_saved_scheme_selectors(self, base_name=None, current_name=None):
        names = [scheme['name'] for scheme in self.saved_schemes]
        for combo in [self.cb_compare_base, self.cb_compare_current]:
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            combo.blockSignals(False)
        if names:
            if len(names) >= 2:
                self.cb_compare_base.setCurrentText(base_name if base_name in names else names[-2])
                self.cb_compare_current.setCurrentText(current_name if current_name in names else names[-1])
            else:
                self.cb_compare_base.setCurrentText(names[0])
                self.cb_compare_current.setCurrentText(names[0])
        self.update_comparison_panel()

    def update_comparison_panel(self, *args):
        if len(self.saved_schemes) < 2:
            self.clear_comparison_panel("请至少保存两个方案后再进行性能对比。")
            return
        base_name = self.cb_compare_base.currentText().strip()
        current_name = self.cb_compare_current.currentText().strip()
        if not base_name or not current_name:
            self.clear_comparison_panel("请选择两个已保存方案。")
            return
        if base_name == current_name:
            self.clear_comparison_panel("请选择两个不同的已保存方案进行对比。")
            return
        base_scheme = self.get_saved_scheme(base_name)
        current_scheme = self.get_saved_scheme(current_name)
        if not base_scheme or not current_scheme:
            self.clear_comparison_panel("方案数据不存在，请重新选择。")
            return

        self.lbl_compare_hint.setText(
            f"变化值 = 对比方案 - 基准方案 | {base_name} vs {current_name}"
        )
        for key in TARGET_KEYS:
            base_val = float(base_scheme['results'][key])
            current_val = float(current_scheme['results'][key])
            delta_val = current_val - base_val
            if abs(delta_val) < 1e-9:
                delta_color = "#7f8c8d"
            elif delta_val > 0:
                delta_color = "#d35400"
            else:
                delta_color = "#2980b9"
            self.compare_labels[key]['base'].setText(self.format_metric(key, base_val))
            self.compare_labels[key]['current'].setText(self.format_metric(key, current_val))
            self.compare_labels[key]['delta'].setText(f"{delta_val:+.2f}")
            self.compare_labels[key]['delta'].setStyleSheet(f"font-weight: bold; color: {delta_color};")

    def save_current_scheme(self):
        if not self.last_prediction:
            QMessageBox.information(self, "暂无可保存方案", "请先执行一次预测，再保存当前设计方案。")
            return
        default_name = (
            f"方案{len(self.saved_schemes) + 1} - "
            f"{self.last_prediction['ori'].split()[0]} / {self.last_prediction['shade'].split()[0]}"
        )
        scheme_name, ok = QInputDialog.getText(self, "保存设计方案", "请输入方案名称：", text=default_name)
        if not ok:
            return
        scheme_name = self.make_scheme_name_unique((scheme_name or default_name).strip() or default_name)
        previous_name = self.saved_schemes[-1]['name'] if self.saved_schemes else None
        self.saved_schemes.append({
            'name': scheme_name,
            'ori': self.last_prediction['ori'],
            'shade': self.last_prediction['shade'],
            'raw_inputs': dict(self.last_prediction['raw_inputs']),
            'calculated': dict(self.last_prediction['calculated']),
            'results': dict(self.last_prediction['results'])
        })
        base_name = previous_name if previous_name else scheme_name
        self.refresh_saved_scheme_selectors(base_name=base_name, current_name=scheme_name)
        if len(self.saved_schemes) == 1:
            msg = f"{scheme_name} 已保存。继续保存新的方案后，就可以在右侧直接做性能对比。"
        else:
            msg = f"{scheme_name} 已保存，并已与上一版方案建立对比。"
        QMessageBox.information(self, "方案已保存", msg)

    def apply_presets(self):
        ori, sha = self.cb_ori.currentText(), self.cb_sha.currentText()
        preset = PRESET_DATA.get((ori, sha), {})
        for feat, val in preset.items():
            if feat in self.inputs:
                self.set_feature_value(feat, float(val))

        active_list = SHADE_FEATURE_MAP.get(sha, FEATURE_ORDER)
        is_p = "垂直" in sha or "组合" in sha
        for f, (lbl, row_widget, val_lbl) in self.rows.items():
            vis = f in active_list or (f == 'WWR_s_exp' and is_p)
            lbl.setVisible(vis)
            row_widget.setVisible(vis)
            if f == 'WWR_s':
                self.inputs[f].setEnabled(not is_p)

    def reset_3d_view(self):
        self.ax.view_init(elev=25, azim=45)
        self.canvas.draw()

    def update_viz_only(self):
        sha_mode = self.cb_sha.currentText()
        raw_in = {f: self.get_feature_value(f) for f in FEATURE_ORDER}
        cal_data = geometry_engine(raw_in, sha_mode)
        saved_elev, saved_azim = self.ax.elev, self.ax.azim
        draw_classroom_3d(self.ax, cal_data, sha_mode, self.chk_upper.isChecked(), self.chk_side.isChecked())
        self.ax.view_init(elev=saved_elev, azim=saved_azim)
        self.canvas.draw()

    def export_3d_model(self):
        sha = self.cb_sha.currentText()
        raw = {f: self.get_feature_value(f) for f in FEATURE_ORDER}
        cal = geometry_engine(raw, sha)
        fmt = self.cb_export_fmt.currentText()
        if "3DS" in fmt:
            path, _ = QFileDialog.getSaveFileName(
                self, "导出 3D 模型", "classroom_model.3ds", "3DS Files (*.3ds)")
            if not path:
                return
            try:
                out = export_classroom_3ds(path, cal, sha)
                QMessageBox.information(self, "导出成功",
                    f"3DS 模型已导出（按图例分层）：\n\n"
                    f"模型文件：{out}\n\n"
                    f"可直接导入 3ds Max / Blender / Rhino 等软件。\n"
                    f"各图层在导入后将自动显示为独立对象。")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"导出过程出错：\n{str(e)}")
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "导出 3D 模型", "classroom_model.obj", "OBJ Files (*.obj)")
            if not path:
                return
            try:
                obj_path, mtl_path = export_classroom_obj(path, cal, sha)
                QMessageBox.information(self, "导出成功",
                    f"OBJ 模型已导出（按图例分层）：\n\n"
                    f"模型文件：{obj_path}\n材质文件：{mtl_path}\n\n"
                    f"可直接导入 3ds Max / Blender / Rhino / SketchUp 等软件。")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"导出过程出错：\n{str(e)}")

    def do_prediction(self):
        ori_mode = self.cb_ori.currentText()
        sha_mode = self.cb_sha.currentText()
        folder = self.model_map.get((ori_mode, sha_mode))
        if not folder:
            QMessageBox.warning(self, "模型缺失", "当前朝向与遮阳形式尚未找到对应模型目录。")
            return

        self.last_prediction = None
        self.btn_save_scheme.setEnabled(False)

        try:
            self.btn_run.setEnabled(False)
            QApplication.processEvents()

            raw_in = {f: self.get_feature_value(f) for f in FEATURE_ORDER}
            cal_data = geometry_engine(raw_in, sha_mode)

            if "悬挑" in sha_mode or "组合" in sha_mode:
                alpha_val = self.get_feature_value('alpha_oh')
                if alpha_val < 0:
                    shs_val = self.get_feature_value('SH_s')
                    dmv_val = self.get_feature_value('d_mv')
                    max_L_oh = round(dmv_val + shs_val, 2)
                    input_L_oh = self.get_feature_value('L_oh')
                    if input_L_oh > max_L_oh:
                        QMessageBox.warning(
                            self, "遮阳板长度约束",
                            f"当 αoh < 0 时，Loh 必须 ≤ dmv + SHs\n\n"
                            f"  当前 dmv = {dmv_val},  SHs = {shs_val}\n"
                            f"  最大允许值 = {max_L_oh}\n"
                            f"  您设定的 Loh = {input_L_oh}\n\n"
                            f"Loh 已自动限制为 {cal_data['L_oh']}"
                        )

            if "垂直" in sha_mode or "组合" in sha_mode:
                input_mtl = raw_in.get('M_TL', 0.0)
                input_mtr = raw_in.get('M_TR', 0.0)
                wv_val = raw_in.get('W_v', 0.01)
                if input_mtl + input_mtr > wv_val:
                    QMessageBox.warning(
                        self, "垂直遮阳移动约束",
                        f"MTL + MTR 之和不能超过 Wv\n\n"
                        f"  当前 Wv = {wv_val}\n"
                        f"  您设定的 MTL = {input_mtl},  MTR = {input_mtr}\n"
                        f"  合计 = {round(input_mtl + input_mtr, 2)}  >  Wv = {wv_val}\n\n"
                        f"已按比例自动缩放：MTL = {cal_data['M_TL']},  MTR = {cal_data['M_TR']}"
                    )

            writeback_keys = [
                'WWR_c', 'WWR_s', 'H_s', 'W_s', 'W_c',
                'SH_s', 'SH_c', 'd_A_s', 'd_B_s', 'd_A_c', 'd_B_c',
                'L_oh', 'd_mv', 'M_TL', 'M_TR'
            ]
            for key in writeback_keys:
                if key in cal_data:
                    self.set_feature_value(key, cal_data[key])

            active_feats = SHADE_FEATURE_MAP.get(sha_mode, FEATURE_ORDER[1:13])
            X = pd.DataFrame([[cal_data[f] for f in active_feats]], columns=active_feats)

            ckey = (ori_mode, sha_mode)
            if ckey not in self.model_cache:
                self.model_cache[ckey] = {
                    f"{t}_{m}": joblib.load(os.path.join(folder, f"{m}_model_{t}.joblib"))
                    for t in TARGET_KEYS for m in ["xgb", "lgbm", "rf", "meta"]
                }

            mc = self.model_cache[ckey]
            result_values = {}
            for t in TARGET_KEYS:
                m1 = mc[f"{t}_xgb"].predict(X)
                m2 = mc[f"{t}_lgbm"].predict(X)
                m3 = mc[f"{t}_rf"].predict(X)
                final = mc[f"{t}_meta"].predict(np.column_stack([m1, m2, m3]))[0]

                if t in ["sDA", "sGA"]:
                    final = max(0.0, min(final, 1.0))
                    v_show = final * 100
                elif t in ["UDI", "SVF"]:
                    v_show = max(0.0, min(final, 100.0))
                else:
                    v_show = final

                result_values[t] = float(v_show)
                self.res_labels[t].setText(f"{v_show:.2f}")

            saved_elev, saved_azim = self.ax.elev, self.ax.azim
            draw_classroom_3d(self.ax, cal_data, sha_mode, self.chk_upper.isChecked(), self.chk_side.isChecked())
            self.ax.view_init(elev=saved_elev, azim=saved_azim)
            self.canvas.draw()

            self.last_prediction = {
                'ori': ori_mode, 'shade': sha_mode,
                'raw_inputs': dict(raw_in), 'calculated': dict(cal_data),
                'results': result_values
            }
            self.btn_save_scheme.setEnabled(True)
            self.btn_run.setEnabled(True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "运行故障", f"预测或计算出错：\n{str(e)}")
            self.btn_save_scheme.setEnabled(False)
            self.btn_run.setEnabled(True)


# %% [Cell 4F: W_v 特殊刻度 + 约束提示显示同步 + 预测视角保持]

FEATURE_META['W_v']['step'] = 0.05
FEATURE_META['W_v']['slider_values'] = [0.01] + [round(i * 0.05, 2) for i in range(1, 25)]


def _wv_slider_values():
    return FEATURE_META['W_v']['slider_values']


def _nearest_wv_index(real_val):
    values = _wv_slider_values()
    target = float(real_val)
    return min(range(len(values)), key=lambda idx: abs(values[idx] - target))


def _format_feature_value(feat, value):
    meta = FEATURE_META[feat]
    if feat == 'W_v':
        return f"{float(value):.2f} {meta['unit']}"

    step_text = str(meta['step'])
    decimals = len(step_text.rstrip('0').split('.')[1]) if '.' in step_text else 0
    return f"{float(value):.{decimals}f} {meta['unit']}"


def _scale_vertical_offsets_to_slider(mtl, mtr, wv_limit):
    step = float(FEATURE_META['M_TL']['step'])
    limit = round(max(0.0, float(wv_limit)), 2)
    total = max(0.0, float(mtl) + float(mtr))

    if total <= limit + 1e-9:
        return round(float(mtl), 2), round(float(mtr), 2)

    if limit < step:
        return 0.0, 0.0

    ratio = limit / max(total, 1e-9)
    scaled = [float(mtl) * ratio, float(mtr) * ratio]
    snapped = [round(round(val / step) * step, 2) for val in scaled]

    while snapped[0] + snapped[1] > limit + 1e-9:
        idx = 0 if snapped[0] >= snapped[1] else 1
        if snapped[idx] >= step:
            snapped[idx] = round(snapped[idx] - step, 2)
        elif snapped[1 - idx] >= step:
            snapped[1 - idx] = round(snapped[1 - idx] - step, 2)
        else:
            return 0.0, 0.0

    return max(0.0, snapped[0]), max(0.0, snapped[1])


def geometry_engine(ui, mode):
    wa = 27.0
    wl = 9.0
    wall_h = 3.8
    min_top_clearance = 0.8

    def safe_float(key, default=0.0):
        val = ui.get(key, default)
        try:
            return float(val) if val is not None else default
        except Exception:
            return default

    def skylight_height_cap(wwr_value):
        if abs(wwr_value - 30.0) < 1e-9:
            return 2.0
        if abs(wwr_value - 35.0) < 1e-9:
            return 2.2
        return 2.6

    def adjust_skylight_sill(hs_value, shs_value):
        shs_actual = min(max(round(shs_value, 2), 0.4), 1.0)
        top_gap = round(wall_h - hs_value - shs_actual, 2)

        while top_gap < min_top_clearance and shs_actual > 0.4:
            shs_actual = round(max(0.4, shs_actual - 0.1), 2)
            top_gap = round(wall_h - hs_value - shs_actual, 2)

        if top_gap < 0.05:
            top_gap = 0.0
        return round(shs_actual, 2), round(top_gap, 2)

    hs_in = safe_float('H_s', 2.0)
    hc = safe_float('H_c', 1.0)
    shs_in = safe_float('SH_s', 0.9)
    shc_in = safe_float('SH_c', 0.8)
    wv = safe_float('W_v', 0.1)

    L_orig = safe_float('L_oh', 1.0)
    d_orig = safe_float('d_mv', 0.5)
    angle = safe_float('alpha_oh', 0.0)

    L_final = L_orig
    d_final = d_orig
    if angle < 0:
        max_L_oh = round(d_final + shs_in, 2)
        if L_final > max_L_oh:
            L_final = max_L_oh
            if L_final < 0:
                L_final = 0

    wwr_c_in = safe_float('WWR_c', 15)
    max_area_c = 6.2 * hc
    max_wwr_c = (max_area_c / wa) * 100.0
    real_wwr_c = min(wwr_c_in, max_wwr_c)
    wc_actual = min((real_wwr_c / 100.0 * wa) / max(0.1, hc), 6.2)

    if hc <= 0.8:
        shc_actual = 2.0
    else:
        shc_actual = min(max(shc_in, 0.8), 1.0)

    is_vertical_mode = mode in ["垂直 (Vertical)", "组合 (O+V)"]
    design_wwr_s = safe_float('WWR_s_exp', 30) if is_vertical_mode else safe_float('WWR_s', 30)
    hs_cap = skylight_height_cap(design_wwr_s)
    hs_actual = min(max(round(hs_in, 1), 1.2), hs_cap)
    target_area_s = round((design_wwr_s / 100.0) * wa, 3)

    total_opening_w = round(target_area_s / max(hs_actual, 0.1), 3)
    while total_opening_w >= wl and hs_actual < hs_cap:
        hs_actual = round(min(hs_actual + 0.1, hs_cap), 1)
        total_opening_w = round(target_area_s / max(hs_actual, 0.1), 3)

    total_opening_w = min(total_opening_w, wl)

    if is_vertical_mode:
        nv = max(2, int(safe_float('N_v', 2)))
        net_glass_w_theory = max(0.1, total_opening_w - (nv - 1) * wv)
        ws_pane = round(net_glass_w_theory / max(1, nv - 1), 1)
        net_glass_w_actual = ws_pane * (nv - 1)
        actual_wwr_s = round((net_glass_w_actual * hs_actual / wa) * 100.0)
        total_opening_w_actual = net_glass_w_actual + (nv - 1) * wv
    else:
        nv = 0
        ws_pane = round(total_opening_w, 1)
        actual_wwr_s = round((ws_pane * hs_actual / wa) * 100.0)
        total_opening_w_actual = ws_pane

    shs_actual, _ = adjust_skylight_sill(hs_actual, shs_in)

    total_s = max(0.0, round(wl - total_opening_w_actual, 1))
    d_a_s_actual, d_b_s_actual = _resolve_offsets_with_mins(
        safe_float('d_A_s', MIN_D_A_S),
        safe_float('d_B_s', MIN_D_B_S),
        total_s, MIN_D_A_S, MIN_D_B_S
    )

    total_c = max(0.0, round(CORRIDOR_USABLE - wc_actual, 1))
    d_a_c_actual, d_b_c_actual = _resolve_offsets_with_mins(
        safe_float('d_A_c', total_c / 2.0),
        safe_float('d_B_c', total_c / 2.0),
        total_c, 0.1, 0.1
    )

    mtl = safe_float('M_TL', 0.0)
    mtr = safe_float('M_TR', 0.0)
    mtl, mtr = _scale_vertical_offsets_to_slider(mtl, mtr, wv)

    res = {k: safe_float(k) for k in FEATURE_ORDER}
    res.update({
        'H_s': hs_actual,
        'WWR_s': int(actual_wwr_s),
        'WWR_c': int(real_wwr_c),
        'W_s': ws_pane,
        'W_c': round(wc_actual, 2),
        'SH_s': shs_actual,
        'SH_c': round(shc_actual, 2),
        'd_A_s': d_a_s_actual,
        'd_B_s': d_b_s_actual,
        'd_A_c': d_a_c_actual,
        'd_B_c': d_b_c_actual,
        'N_v': nv,
        'M_TL': mtl,
        'M_TR': mtr,
        'L_oh': L_final,
        'd_mv': d_final
    })
    return res


BaseClassroomPredictorApp_V2 = ClassroomPredictorApp


class ClassroomPredictorApp(BaseClassroomPredictorApp_V2):
    def init_ui(self):
        super().init_ui()
        if 'W_v' in self.inputs:
            slider = self.inputs['W_v']
            _, _, val_lbl = self.rows['W_v']

            def sync_wv_label(v):
                if val_lbl is None:
                    return
                idx = max(0, min(int(v), len(_wv_slider_values()) - 1))
                val_lbl.setText(_format_feature_value('W_v', _wv_slider_values()[idx]))

            slider.valueChanged.connect(sync_wv_label)
            sync_wv_label(slider.value())

    def get_feature_value(self, feat):
        if feat == 'W_v':
            slider = self.inputs[feat]
            idx = max(0, min(slider.value(), len(_wv_slider_values()) - 1))
            return float(_wv_slider_values()[idx])
        return super().get_feature_value(feat)

    def set_feature_value(self, feat, real_val):
        if feat == 'W_v':
            slider = self.inputs[feat]
            idx = _nearest_wv_index(real_val)
            slider.blockSignals(True)
            slider.setValue(idx)
            slider.blockSignals(False)
            _, _, val_lbl = self.rows[feat]
            if val_lbl:
                val_lbl.setText(_format_feature_value(feat, _wv_slider_values()[idx]))
            return
        return super().set_feature_value(feat, real_val)

    def capture_3d_view_state(self):
        return {
            'elev': self.ax.elev,
            'azim': self.ax.azim,
            'roll': getattr(self.ax, 'roll', 0.0),
            'xlim': self.ax.get_xlim3d(),
            'ylim': self.ax.get_ylim3d(),
            'zlim': self.ax.get_zlim3d(),
        }

    def restore_3d_view_state(self, state, restore_limits=False):
        if not state:
            return
        try:
            self.ax.view_init(elev=state.get('elev', 25), azim=state.get('azim', 45), roll=state.get('roll', 0.0))
        except TypeError:
            self.ax.view_init(elev=state.get('elev', 25), azim=state.get('azim', 45))

        if restore_limits:
            self.ax.set_xlim3d(*state['xlim'])
            self.ax.set_ylim3d(*state['ylim'])
            self.ax.set_zlim3d(*state['zlim'])

    def update_viz_only(self):
        sha_mode = self.cb_sha.currentText()
        raw_in = {f: self.get_feature_value(f) for f in FEATURE_ORDER}
        cal_data = geometry_engine(raw_in, sha_mode)
        saved_view = self.capture_3d_view_state()
        draw_classroom_3d(self.ax, cal_data, sha_mode, self.chk_upper.isChecked(), self.chk_side.isChecked())
        self.restore_3d_view_state(saved_view, restore_limits=False)
        self.canvas.draw()

    def do_prediction(self):
        ori_mode = self.cb_ori.currentText()
        sha_mode = self.cb_sha.currentText()
        folder = self.model_map.get((ori_mode, sha_mode))
        if not folder:
            QMessageBox.warning(self, "模型缺失", "当前朝向与遮阳形式尚未找到对应模型目录。")
            return

        self.last_prediction = None
        self.btn_save_scheme.setEnabled(False)

        try:
            self.btn_run.setEnabled(False)
            QApplication.processEvents()

            raw_in = {f: self.get_feature_value(f) for f in FEATURE_ORDER}
            cal_data = geometry_engine(raw_in, sha_mode)

            if "悬挑" in sha_mode or "组合" in sha_mode:
                alpha_val = self.get_feature_value('alpha_oh')
                if alpha_val < 0:
                    shs_val = self.get_feature_value('SH_s')
                    dmv_val = self.get_feature_value('d_mv')
                    max_L_oh = round(dmv_val + shs_val, 2)
                    input_L_oh = self.get_feature_value('L_oh')
                    if input_L_oh > max_L_oh:
                        QMessageBox.warning(
                            self, "遮阳板长度约束",
                            f"当 αoh < 0 时，Loh 必须 ≤ dmv + SHs\n\n"
                            f"  当前 dmv = {dmv_val},  SHs = {shs_val}\n"
                            f"  最大允许值 = {max_L_oh}\n"
                            f"  您设定的 Loh = {input_L_oh}\n\n"
                            f"Loh 已自动限制为 {cal_data['L_oh']}"
                        )

            if "垂直" in sha_mode or "组合" in sha_mode:
                input_mtl = raw_in.get('M_TL', 0.0)
                input_mtr = raw_in.get('M_TR', 0.0)
                wv_val = raw_in.get('W_v', 0.01)
                if input_mtl + input_mtr > wv_val:
                    QMessageBox.warning(
                        self, "垂直遮阳移动约束",
                        f"MTL + MTR 之和不能超过 Wv\n\n"
                        f"  当前 Wv = {_format_feature_value('W_v', wv_val)}\n"
                        f"  您设定的 MTL = {_format_feature_value('M_TL', input_mtl)},  MTR = {_format_feature_value('M_TR', input_mtr)}\n"
                        f"  合计 = {input_mtl + input_mtr:.2f} m  >  Wv = {_format_feature_value('W_v', wv_val)}\n\n"
                        f"已按比例自动缩放：MTL = {_format_feature_value('M_TL', cal_data['M_TL'])},  MTR = {_format_feature_value('M_TR', cal_data['M_TR'])}"
                    )

            writeback_keys = [
                'WWR_c', 'WWR_s', 'H_s', 'W_s', 'W_c',
                'SH_s', 'SH_c', 'd_A_s', 'd_B_s', 'd_A_c', 'd_B_c',
                'L_oh', 'd_mv', 'M_TL', 'M_TR'
            ]
            for key in writeback_keys:
                if key in cal_data:
                    self.set_feature_value(key, cal_data[key])

            active_feats = SHADE_FEATURE_MAP.get(sha_mode, FEATURE_ORDER[1:13])
            X = pd.DataFrame([[cal_data[f] for f in active_feats]], columns=active_feats)

            ckey = (ori_mode, sha_mode)
            if ckey not in self.model_cache:
                self.model_cache[ckey] = {
                    f"{t}_{m}": joblib.load(os.path.join(folder, f"{m}_model_{t}.joblib"))
                    for t in TARGET_KEYS for m in ["xgb", "lgbm", "rf", "meta"]
                }

            mc = self.model_cache[ckey]
            result_values = {}
            for t in TARGET_KEYS:
                m1 = mc[f"{t}_xgb"].predict(X)
                m2 = mc[f"{t}_lgbm"].predict(X)
                m3 = mc[f"{t}_rf"].predict(X)
                final = mc[f"{t}_meta"].predict(np.column_stack([m1, m2, m3]))[0]

                if t in ["sDA", "sGA"]:
                    final = max(0.0, min(final, 1.0))
                    v_show = final * 100
                elif t in ["UDI", "SVF"]:
                    v_show = max(0.0, min(final, 100.0))
                else:
                    v_show = final

                result_values[t] = float(v_show)
                self.res_labels[t].setText(f"{v_show:.2f}")

            saved_view = self.capture_3d_view_state()
            draw_classroom_3d(self.ax, cal_data, sha_mode, self.chk_upper.isChecked(), self.chk_side.isChecked())
            self.restore_3d_view_state(saved_view, restore_limits=True)
            self.canvas.draw()

            self.last_prediction = {
                'ori': ori_mode,
                'shade': sha_mode,
                'raw_inputs': dict(raw_in),
                'calculated': dict(cal_data),
                'results': result_values
            }
            self.btn_save_scheme.setEnabled(True)
            self.btn_run.setEnabled(True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "运行故障", f"预测或计算出错：\n{str(e)}")
            self.btn_save_scheme.setEnabled(False)
            self.btn_run.setEnabled(True)


# %% [Cell 4D: 几何与绘图修正 - 支持窗口偏移]

# 走廊侧物理常量：两扇门各占 0.2m 偏移 + 1.2m 门宽 = 1.4m
DOOR_ZONE = 1.4          # 每扇门占的 y 范围 (0→1.4 和 7.6→9.0)
CORRIDOR_USABLE = 9.0 - DOOR_ZONE * 2   # = 6.2 m

# 采光窗侧物理常量
MIN_D_A_S = 1.0   # 距黑板侧最小间距
MIN_D_B_S = 0.2   # 另一侧最小间距


def _resolve_offsets_with_mins(left_in, right_in, total_avail, min_left, min_right):
    """在 min_left / min_right 约束下分配 left 和 right，使 left+right = total_avail。"""
    total_avail = max(0.0, round(float(total_avail), 1))
    min_left = round(float(min_left), 1)
    min_right = round(float(min_right), 1)
    left_in = round(float(left_in), 1)
    right_in = round(float(right_in), 1)

    if total_avail <= 0:
        return min_left, min_right

    left_in = max(min_left, left_in)
    right_in = max(min_right, right_in)

    free = max(0.0, total_avail - min_left - min_right)
    if free <= 0:
        return round(min_left, 1), round(total_avail - min_left, 1)

    excess_left = left_in - min_left
    excess_right = right_in - min_right
    excess_total = excess_left + excess_right

    if excess_total <= 1e-9:
        fl = round(free / 2.0, 1)
        fr = round(free - fl, 1)
    else:
        fl = round(excess_left / excess_total * free, 1)
        fr = round(free - fl, 1)

    return round(min_left + fl, 1), round(min_right + fr, 1)


def geometry_engine(ui, mode):
    wa = 27.0
    wl = 9.0
    wall_h = 3.8
    min_top_clearance = 0.8

    def safe_float(key, default=0.0):
        val = ui.get(key, default)
        try:
            return float(val) if val is not None else default
        except Exception:
            return default

    def skylight_height_cap(wwr_value):
        if abs(wwr_value - 30.0) < 1e-9:
            return 2.0
        if abs(wwr_value - 35.0) < 1e-9:
            return 2.2
        return 2.6

    def adjust_skylight_sill(hs_value, shs_value):
        shs_actual = min(max(round(shs_value, 2), 0.4), 1.0)
        top_gap = round(wall_h - hs_value - shs_actual, 2)

        while top_gap < min_top_clearance and shs_actual > 0.4:
            shs_actual = round(max(0.4, shs_actual - 0.1), 2)
            top_gap = round(wall_h - hs_value - shs_actual, 2)

        if top_gap < 0.05:
            top_gap = 0.0
        return round(shs_actual, 2), round(top_gap, 2)

    hs_in = safe_float('H_s', 2.0)
    hc = safe_float('H_c', 1.0)
    shs_in = safe_float('SH_s', 0.9)
    shc_in = safe_float('SH_c', 0.8)
    wv = safe_float('W_v', 0.1)

    L_orig = safe_float('L_oh', 1.0)
    d_orig = safe_float('d_mv', 0.5)
    angle = safe_float('alpha_oh', 0.0)

    L_final = L_orig
    d_final = d_orig
    if angle < 0:
        max_L_oh = round(d_final + shs_in, 2)
        if L_final > max_L_oh:
            L_final = max_L_oh
            if L_final < 0:
                L_final = 0

    wwr_c_in = safe_float('WWR_c', 15)
    max_area_c = 6.2 * hc
    max_wwr_c = (max_area_c / wa) * 100.0
    real_wwr_c = min(wwr_c_in, max_wwr_c)
    wc_actual = min((real_wwr_c / 100.0 * wa) / max(0.1, hc), 6.2)

    if hc <= 0.8:
        shc_actual = 2.0
    else:
        shc_actual = min(max(shc_in, 0.8), 1.0)

    is_vertical_mode = mode in ["垂直 (Vertical)", "组合 (O+V)"]
    design_wwr_s = safe_float('WWR_s_exp', 30) if is_vertical_mode else safe_float('WWR_s', 30)
    hs_cap = skylight_height_cap(design_wwr_s)
    hs_actual = min(max(round(hs_in, 1), 1.2), hs_cap)
    target_area_s = round((design_wwr_s / 100.0) * wa, 3)

    total_opening_w = round(target_area_s / max(hs_actual, 0.1), 3)
    while total_opening_w >= wl and hs_actual < hs_cap:
        hs_actual = round(min(hs_actual + 0.1, hs_cap), 1)
        total_opening_w = round(target_area_s / max(hs_actual, 0.1), 3)

    total_opening_w = min(total_opening_w, wl)

    if is_vertical_mode:
        nv = max(2, int(safe_float('N_v', 2)))
        net_glass_w_theory = max(0.1, total_opening_w - (nv - 1) * wv)
        ws_pane = round(net_glass_w_theory / max(1, nv - 1), 1)
        net_glass_w_actual = ws_pane * (nv - 1)
        actual_wwr_s = round((net_glass_w_actual * hs_actual / wa) * 100.0)
        total_opening_w_actual = net_glass_w_actual + (nv - 1) * wv
    else:
        nv = 0
        ws_pane = round(total_opening_w, 1)
        actual_wwr_s = round((ws_pane * hs_actual / wa) * 100.0)
        total_opening_w_actual = ws_pane

    shs_actual, _ = adjust_skylight_sill(hs_actual, shs_in)

    # 采光窗偏移：d_A_s >= 1.0 (离黑板), d_B_s >= 0.2
    total_s = max(0.0, round(wl - total_opening_w_actual, 1))
    d_a_s_actual, d_b_s_actual = _resolve_offsets_with_mins(
        safe_float('d_A_s', MIN_D_A_S),
        safe_float('d_B_s', MIN_D_B_S),
        total_s, MIN_D_A_S, MIN_D_B_S
    )

    # 走廊窗偏移：可用空间 = 6.2 - W_c，两侧最小 0.1
    total_c = max(0.0, round(CORRIDOR_USABLE - wc_actual, 1))
    d_a_c_actual, d_b_c_actual = _resolve_offsets_with_mins(
        safe_float('d_A_c', total_c / 2.0),
        safe_float('d_B_c', total_c / 2.0),
        total_c, 0.1, 0.1
    )

    # --- M_TL + M_TR ≤ W_v 约束 ---
    mtl = safe_float('M_TL', 0.0)
    mtr = safe_float('M_TR', 0.0)
    if mtl + mtr > wv:
        ratio = wv / max(mtl + mtr, 1e-9)
        mtl = round(mtl * ratio, 2)
        mtr = round(mtr * ratio, 2)

    res = {k: safe_float(k) for k in FEATURE_ORDER}
    res.update({
        'H_s': hs_actual,
        'WWR_s': int(actual_wwr_s),
        'WWR_c': int(real_wwr_c),
        'W_s': ws_pane,
        'W_c': round(wc_actual, 2),
        'SH_s': shs_actual,
        'SH_c': round(shc_actual, 2),
        'd_A_s': d_a_s_actual,
        'd_B_s': d_b_s_actual,
        'd_A_c': d_a_c_actual,
        'd_B_c': d_b_c_actual,
        'N_v': nv,
        'M_TL': mtl,
        'M_TR': mtr,
        'L_oh': L_final,
        'd_mv': d_final
    })
    return res


def draw_classroom_3d(ax, data, sha_mode, show_upper=False, show_side=False, facade_filter=None):
    ax.clear()
    L, D, H_total = 9.0, 9.0, 3.8

    ax.set_proj_type('ortho')
    span = 14
    ax.set_axis_off()
    hide_corridor = facade_filter == 'daylight_front'
    hide_daylight = facade_filter == 'corridor_back'

    def render_module(dy=0, dz=0, is_main=True):
        fac_boost = facade_filter is not None and is_main
        alpha_wall = 0.4 if is_main else 0.1
        alpha_solid = 0.8 if is_main else 0.15
        alpha_glass = 0.5 if is_main else 0.15
        if fac_boost:
            alpha_wall = min(0.85, alpha_wall + 0.35)
            alpha_solid = min(0.98, alpha_solid + 0.12)
            alpha_glass = min(0.9, alpha_glass + 0.38)
        lw_main = 1.5 if is_main else 0.5
        c_edge = 'k' if is_main else 'gray'

        for z in [0, H_total]:
            ax.plot3D([0, 9, 9, 0, 0], [0 + dy, 0 + dy, 9 + dy, 9 + dy, 0 + dy], [z + dz, z + dz, z + dz, z + dz, z + dz], c_edge, lw=lw_main)
        for x, y in [(0, 0), (9, 0), (9, 9), (0, 9)]:
            ax.plot3D([x, x], [y + dy, y + dy], [0 + dz, H_total + dz], c_edge, lw=1, alpha=alpha_wall)

        if not hide_corridor:
            ax.add_collection3d(Poly3DCollection([[(-3, 0 + dy, 0 + dz), (0, 0 + dy, 0 + dz), (0, 9 + dy, 0 + dz), (-3, 9 + dy, 0 + dz)]], facecolors='gray', alpha=0.1))
            c_plenum = [
                [(-3, 0 + dy, 3.0 + dz), (0, 0 + dy, 3.0 + dz), (0, 9 + dy, 3.0 + dz), (-3, 9 + dy, 3.0 + dz)],
                [(-3, 0 + dy, 3.8 + dz), (0, 0 + dy, 3.8 + dz), (0, 9 + dy, 3.8 + dz), (-3, 9 + dy, 3.8 + dz)]
            ]
            ax.add_collection3d(Poly3DCollection(c_plenum, facecolors='gray', alpha=0.3))
            wc_draw = data.get('W_c', 4.0)
            hc_draw = data.get('H_c', 1.0)
            shc_draw = data.get('SH_c', 1.0)
            d_a_c = data.get('d_A_c', 0.0)
            yc_start_c = DOOR_ZONE + d_a_c
            win_c_verts = [[
                (0, yc_start_c + dy, shc_draw + dz),
                (0, yc_start_c + wc_draw + dy, shc_draw + dz),
                (0, yc_start_c + wc_draw + dy, shc_draw + hc_draw + dz),
                (0, yc_start_c + dy, shc_draw + hc_draw + dz)
            ]]
            ax.add_collection3d(Poly3DCollection(win_c_verts, facecolors='lightgreen', edgecolors='g', alpha=alpha_glass, lw=1))
            for yd in [0.2, 9 - 0.2 - 1.2]:
                dr = [[(0, yd + dy, 0 + dz), (0, yd + 1.2 + dy, 0 + dz), (0, yd + 1.2 + dy, 2.2 + dz), (0, yd + dy, 2.2 + dz)]]
                ax.add_collection3d(Poly3DCollection(dr, facecolors='saddlebrown', edgecolors=c_edge, alpha=alpha_solid))

        board_y = 0.02 + dy
        board_x1, board_x2 = 2.0, 7.0
        board_z1, board_z2 = 0.9, 2.3
        board_verts = [[
            (board_x1, board_y, board_z1 + dz), (board_x2, board_y, board_z1 + dz),
            (board_x2, board_y, board_z2 + dz), (board_x1, board_y, board_z2 + dz)
        ]]
        ax.add_collection3d(Poly3DCollection(board_verts, facecolors='#243b2f', edgecolors='black', alpha=0.9 if is_main else 0.2, lw=1))
        if is_main:
            ax.text((board_x1 + board_x2) / 2, board_y, board_z2 + 0.15 + dz, '黑板', color='black', ha='center', va='bottom', fontsize=10)

        is_v = "垂直" in sha_mode or "组合" in sha_mode
        nv = int(data.get('N_v', 0))
        wv = data.get('W_v', 0.1)
        ws_pane = data.get('W_s', 4.0)
        hs = data.get('H_s', 2.0)
        shs = data.get('SH_s', 0.9)
        total_opening_w = ((nv - 1) * ws_pane) + (nv - 1) * wv if (is_v and nv >= 2) else ws_pane
        ys_start = data.get('d_A_s', MIN_D_A_S)
        if not hide_daylight:
            num_p = (nv - 1) if (is_v and nv >= 2) else 1
            for i in range(num_p):
                curr_y = ys_start + wv / 2 + i * (ws_pane + wv) if (is_v and nv >= 2) else ys_start
                pane = [[
                    (9, curr_y + dy, shs + dz),
                    (9, curr_y + ws_pane + dy, shs + dz),
                    (9, curr_y + ws_pane + dy, shs + hs + dz),
                    (9, curr_y + dy, shs + hs + dz)
                ]]
                ax.add_collection3d(Poly3DCollection(pane, facecolors='skyblue', edgecolors='b', alpha=alpha_glass))
            if is_v and nv > 0:
                lv, mtl, mtr = data.get('L_v', 0), data.get('M_TL', 0), data.get('M_TR', 0)
                for i in range(nv):
                    by = ys_start + i * (ws_pane + wv)
                    p1, p2, p3, p4 = (9, by - wv / 2 + dy, 0 + dz), (9, by + wv / 2 + dy, 0 + dz), (9, by + wv / 2 + dy, 3.8 + dz), (9, by - wv / 2 + dy, 3.8 + dz)
                    p5, p6, p7, p8 = (9 + lv, by - wv / 2 + mtl + dy, 0 + dz), (9 + lv, by + wv / 2 - mtr + dy, 0 + dz), (9 + lv, by + wv / 2 - mtr + dy, 3.8 + dz), (9 + lv, by - wv / 2 + mtl + dy, 3.8 + dz)
                    ax.add_collection3d(Poly3DCollection([[p1, p2, p3, p4], [p5, p6, p7, p8], [p1, p5, p8, p4], [p2, p6, p7, p3], [p1, p2, p6, p5], [p4, p3, p7, p8]], facecolors='teal', alpha=alpha_solid, edgecolors='k', lw=0.3))
            if ("悬挑" in sha_mode or "组合" in sha_mode) and data.get('L_oh', 0) > 0:
                alpha = np.radians(data.get('alpha_oh', 0))
                loh = data['L_oh']
                dmv = data.get('d_mv', 0)
                zt = 3.8 - dmv
                z_end = (3.8 - dmv) - loh * np.sin(alpha)
                x_ext = 9 + loh * np.cos(alpha)
                oh_verts = [[(9, 0 + dy, zt + dz), (x_ext, 0 + dy, z_end + dz), (x_ext, 9 + dy, z_end + dz), (9, 9 + dy, zt + dz)]]
                ax.add_collection3d(Poly3DCollection(oh_verts, facecolors='indianred', alpha=alpha_solid, edgecolors='darkred', lw=0.5))
            if "框式" in sha_mode:
                fbl, fbr, ftr, ftl = data.get('F_BL', 0), data.get('F_BR', 0), data.get('F_TR', 0), data.get('F_TL', 0)
                p_bl, p_br = (9, ys_start + dy, shs + dz), (9, ys_start + total_opening_w + dy, shs + dz)
                p_tr, p_tl = (9, ys_start + total_opening_w + dy, shs + hs + dz), (9, ys_start + dy, shs + hs + dz)
                e_bl, e_br = (9 + fbl, ys_start + dy, shs + dz), (9 + fbr, ys_start + total_opening_w + dy, shs + dz)
                e_tr, e_tl = (9 + ftr, ys_start + total_opening_w + dy, shs + hs + dz), (9 + ftl, ys_start + dy, shs + hs + dz)
                ax.add_collection3d(Poly3DCollection([[p_bl, p_br, e_br, e_bl], [p_br, p_tr, e_tr, e_br], [p_tr, p_tl, e_tl, e_tr], [p_tl, p_bl, e_bl, e_tl]], facecolors='orange', alpha=alpha_glass, edgecolors='darkorange'))
            if is_main:
                ax.text(9, 4.5 + dy, 4.1 + dz, f"Actual WWR: {data['WWR_s']}%", color='darkred', fontweight='bold', ha='center')

    render_module(dy=0, dz=0, is_main=True)
    if show_upper:
        render_module(dy=0, dz=H_total, is_main=False)
    if show_side:
        render_module(dy=D, dz=0, is_main=False)

    ax.set_xlim(-3.5, 11.5)
    ax.set_ylim(-2.5, 20.5 if show_side else 11.5)
    ax.set_zlim(0, 9 if show_upper else 5)

    # Use the actual axis ranges so the classroom geometry keeps a consistent scale.
    x0, x1 = ax.get_xlim3d()
    y0, y1 = ax.get_ylim3d()
    z0, z1 = ax.get_zlim3d()
    ax.set_box_aspect((abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)))

    legend_handles = []
    if not hide_corridor:
        legend_handles.extend([
            Patch(facecolor='gray', edgecolor='gray', alpha=0.3, label='走廊侧地坪/吊顶'),
            Patch(facecolor='lightgreen', edgecolor='g', alpha=0.5, label='走廊窗'),
            Patch(facecolor='saddlebrown', edgecolor='black', alpha=0.8, label='门'),
        ])
    if not hide_daylight:
        legend_handles.append(Patch(facecolor='skyblue', edgecolor='b', alpha=0.5, label='采光窗'))
    legend_handles.append(Patch(facecolor='#243b2f', edgecolor='black', alpha=0.9, label='黑板'))
    if not hide_daylight:
        if "垂直" in sha_mode or "组合" in sha_mode:
            legend_handles.append(Patch(facecolor='teal', edgecolor='black', alpha=0.8, label='垂直遮阳'))
        if "悬挑" in sha_mode or "组合" in sha_mode:
            legend_handles.append(Patch(facecolor='indianred', edgecolor='darkred', alpha=0.8, label='悬挑遮阳'))
        if "框式" in sha_mode:
            legend_handles.append(Patch(facecolor='orange', edgecolor='darkorange', alpha=0.5, label='框式遮阳'))

    ax.figure.subplots_adjust(right=0.8)
    ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=9, title='颜色说明', title_fontsize=10)


# %% [Cell 4E: GUI 增强覆盖 - 方案三维对比窗口（独立窗口）]
class SchemeComparisonWindow(QMainWindow):
    """独立顶级窗口，与主程序大小一致，标题栏可关闭。"""

    def __init__(self, parent_window=None):
        super().__init__(None)
        self.setWindowTitle("设计方案三维模型对比")
        if parent_window:
            self.resize(parent_window.size())
            self.move(parent_window.pos().x() + 40, parent_window.pos().y() + 40)
        else:
            self.resize(1500, 950)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        header = QLabel("左侧为基准方案  ←→  右侧为对比方案")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50; padding: 6px;")
        layout.addWidget(header)

        canvas_row = QHBoxLayout()
        self.panels = {}
        for key, title_text in [('base', '基准方案'), ('current', '对比方案')]:
            panel_layout = QVBoxLayout()
            title = QLabel(title_text)
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet("font-size: 14px; font-weight: bold; color: #34495e;")
            meta = QLabel("--")
            meta.setAlignment(Qt.AlignCenter)
            meta.setStyleSheet("color: #7f8c8d; font-size: 12px;")

            fig = plt.figure(figsize=(8, 7))
            ax = fig.add_subplot(111, projection='3d')
            canvas = FigureCanvas(fig)

            panel_layout.addWidget(title)
            panel_layout.addWidget(meta)
            panel_layout.addWidget(canvas, 1)
            canvas_row.addLayout(panel_layout)

            self.panels[key] = {
                'meta': meta,
                'figure': fig,
                'ax': ax,
                'canvas': canvas
            }

        layout.addLayout(canvas_row, 1)

    def set_schemes(self, base_scheme, current_scheme):
        for key, scheme in [('base', base_scheme), ('current', current_scheme)]:
            panel = self.panels[key]
            panel['meta'].setText(
                f"{scheme['name']}  |  {scheme['ori']}  |  {scheme['shade']}"
            )
            draw_classroom_3d(panel['ax'], scheme['calculated'], scheme['shade'], False, False)
            panel['ax'].view_init(elev=25, azim=45)
            panel['canvas'].draw()


BaseClassroomPredictorApp = ClassroomPredictorApp


class ClassroomPredictorApp(BaseClassroomPredictorApp):
    def init_ui(self):
        super().init_ui()
        self.compare_window = None
        self.btn_open_compare_window = QPushButton("🧭 打开三维方案对比窗口")
        self.btn_open_compare_window.setFixedHeight(42)
        self.btn_open_compare_window.setEnabled(False)
        self.btn_open_compare_window.setStyleSheet("background-color: #8e44ad; color: white; font-weight: bold;")
        self.btn_open_compare_window.clicked.connect(self.open_comparison_window)

        for group in self.findChildren(QGroupBox):
            if group.title() == "方案保存与性能对比":
                group.layout().addWidget(self.btn_open_compare_window)
                break

    def refresh_compare_controls(self):
        if hasattr(self, 'btn_open_compare_window'):
            self.btn_open_compare_window.setEnabled(len(self.saved_schemes) >= 2)

    def update_comparison_panel(self, *args):
        super().update_comparison_panel(*args)
        self.refresh_compare_controls()
        if self.compare_window and self.compare_window.isVisible():
            self.sync_comparison_window()

    def refresh_saved_scheme_selectors(self, base_name=None, current_name=None):
        super().refresh_saved_scheme_selectors(base_name=base_name, current_name=current_name)
        self.refresh_compare_controls()

    def do_prediction(self):
        super().do_prediction()

    def open_comparison_window(self):
        if len(self.saved_schemes) < 2:
            QMessageBox.information(self, "无法对比", "请至少保存两个方案后，再打开三维模型对比窗口。")
            return

        base_name = self.cb_compare_base.currentText().strip()
        current_name = self.cb_compare_current.currentText().strip()
        if not base_name or not current_name or base_name == current_name:
            QMessageBox.information(self, "无法对比", "请先在右侧选择两个不同的已保存方案。")
            return

        if self.compare_window is None:
            self.compare_window = SchemeComparisonWindow(self)
        else:
            self.compare_window.resize(self.size())
        self.sync_comparison_window()
        self.compare_window.show()
        self.compare_window.raise_()
        self.compare_window.activateWindow()

    def sync_comparison_window(self):
        if self.compare_window is None:
            return
        base_scheme = self.get_saved_scheme(self.cb_compare_base.currentText().strip())
        current_scheme = self.get_saved_scheme(self.cb_compare_current.currentText().strip())
        if not base_scheme or not current_scheme or base_scheme['name'] == current_scheme['name']:
            return
        self.compare_window.set_schemes(base_scheme, current_scheme)


# %% [Cell 4G: 最终覆盖 - W_v 特殊刻度 + 约束提示显示同步 + 预测视角保持]

FEATURE_META['W_v']['step'] = 0.05
FEATURE_META['W_v']['slider_values'] = [0.01] + [round(i * 0.05, 2) for i in range(1, 25)]


def _wv_slider_values():
    return FEATURE_META['W_v']['slider_values']


def _nearest_wv_index(real_val):
    values = _wv_slider_values()
    target = float(real_val)
    return min(range(len(values)), key=lambda idx: abs(values[idx] - target))


def _format_feature_value(feat, value):
    meta = FEATURE_META[feat]
    if feat == 'W_v':
        return f"{float(value):.2f} {meta['unit']}"

    step_text = str(meta['step'])
    decimals = len(step_text.rstrip('0').split('.')[1]) if '.' in step_text else 0
    return f"{float(value):.{decimals}f} {meta['unit']}"


def _scale_vertical_offsets_to_slider(mtl, mtr, wv_limit):
    step = float(FEATURE_META['M_TL']['step'])
    limit = round(max(0.0, float(wv_limit)), 2)
    total = max(0.0, float(mtl) + float(mtr))

    if total <= limit + 1e-9:
        return round(float(mtl), 2), round(float(mtr), 2)

    if limit < step:
        return 0.0, 0.0

    ratio = limit / max(total, 1e-9)
    scaled = [float(mtl) * ratio, float(mtr) * ratio]
    snapped = [round(round(val / step) * step, 2) for val in scaled]

    while snapped[0] + snapped[1] > limit + 1e-9:
        idx = 0 if snapped[0] >= snapped[1] else 1
        if snapped[idx] >= step:
            snapped[idx] = round(snapped[idx] - step, 2)
        elif snapped[1 - idx] >= step:
            snapped[1 - idx] = round(snapped[1 - idx] - step, 2)
        else:
            return 0.0, 0.0

    return max(0.0, snapped[0]), max(0.0, snapped[1])


def geometry_engine(ui, mode):
    wa = 27.0
    wl = 9.0
    wall_h = 3.8
    min_top_clearance = 0.8

    def safe_float(key, default=0.0):
        val = ui.get(key, default)
        try:
            return float(val) if val is not None else default
        except Exception:
            return default

    def skylight_height_cap(wwr_value):
        if abs(wwr_value - 30.0) < 1e-9:
            return 2.0
        if abs(wwr_value - 35.0) < 1e-9:
            return 2.2
        return 2.6

    def adjust_skylight_sill(hs_value, shs_value):
        shs_actual = min(max(round(shs_value, 2), 0.4), 1.0)
        top_gap = round(wall_h - hs_value - shs_actual, 2)

        while top_gap < min_top_clearance and shs_actual > 0.4:
            shs_actual = round(max(0.4, shs_actual - 0.1), 2)
            top_gap = round(wall_h - hs_value - shs_actual, 2)

        if top_gap < 0.05:
            top_gap = 0.0
        return round(shs_actual, 2), round(top_gap, 2)

    hs_in = safe_float('H_s', 2.0)
    hc = safe_float('H_c', 1.0)
    shs_in = safe_float('SH_s', 0.9)
    shc_in = safe_float('SH_c', 0.8)
    wv = safe_float('W_v', 0.1)

    L_orig = safe_float('L_oh', 1.0)
    d_orig = safe_float('d_mv', 0.5)
    angle = safe_float('alpha_oh', 0.0)

    wwr_c_in = safe_float('WWR_c', 15)
    max_area_c = 6.2 * hc
    max_wwr_c = (max_area_c / wa) * 100.0
    real_wwr_c = min(wwr_c_in, max_wwr_c)
    wc_actual = min((real_wwr_c / 100.0 * wa) / max(0.1, hc), 6.2)

    if hc <= 0.8:
        shc_actual = 2.0
    else:
        shc_actual = min(max(shc_in, 0.8), 1.0)

    is_vertical_mode = mode in ["垂直 (Vertical)", "组合 (O+V)"]
    design_wwr_s = safe_float('WWR_s_exp', 30) if is_vertical_mode else safe_float('WWR_s', 30)
    hs_cap = skylight_height_cap(design_wwr_s)
    hs_actual = min(max(round(hs_in, 1), 1.2), hs_cap)
    target_area_s = round((design_wwr_s / 100.0) * wa, 3)
    max_opening_w = round(max(0.0, wl - MIN_D_A_S - MIN_D_B_S), 3)

    total_opening_w = round(target_area_s / max(hs_actual, 0.1), 3)
    while total_opening_w > max_opening_w + 1e-9 and hs_actual < hs_cap:
        hs_actual = round(min(hs_actual + 0.1, hs_cap), 1)
        total_opening_w = round(target_area_s / max(hs_actual, 0.1), 3)

    total_opening_w = min(total_opening_w, max_opening_w)

    if is_vertical_mode:
        nv = max(2, int(safe_float('N_v', 2)))
        net_glass_w_limit = max(0.0, total_opening_w - (nv - 1) * wv)
        ws_pane = round(net_glass_w_limit / max(1, nv - 1), 1)
        while ws_pane > 0 and (ws_pane * (nv - 1) + (nv - 1) * wv) > max_opening_w + 1e-9:
            ws_pane = round(max(0.0, ws_pane - 0.1), 1)
        net_glass_w_actual = ws_pane * (nv - 1)
        total_opening_w_actual = net_glass_w_actual + (nv - 1) * wv
        actual_wwr_s = round((net_glass_w_actual * hs_actual / wa) * 100.0)
    else:
        nv = 0
        ws_pane = round(min(total_opening_w, max_opening_w), 1)
        while ws_pane > max_opening_w + 1e-9:
            ws_pane = round(max(0.0, ws_pane - 0.1), 1)
        total_opening_w_actual = ws_pane
        actual_wwr_s = round((ws_pane * hs_actual / wa) * 100.0)

    shs_actual, _ = adjust_skylight_sill(hs_actual, shs_in)

    max_d_mv = round(max(0.0, 0.8 + (wall_h - 0.8 - hs_actual - shs_actual)), 2)
    d_final = round(min(max(0.0, d_orig), max_d_mv), 2)

    L_final = L_orig
    if angle < 0:
        max_L_oh = round(d_final + shs_actual, 2)
        if L_final > max_L_oh:
            L_final = max_L_oh
            if L_final < 0:
                L_final = 0

    total_s = max(0.0, round(wl - total_opening_w_actual, 1))
    d_a_s_actual, d_b_s_actual = _resolve_offsets_with_mins(
        safe_float('d_A_s', MIN_D_A_S),
        safe_float('d_B_s', MIN_D_B_S),
        total_s, MIN_D_A_S, MIN_D_B_S
    )

    total_c = max(0.0, round(CORRIDOR_USABLE - wc_actual, 1))
    d_a_c_actual, d_b_c_actual = _resolve_offsets_with_mins(
        safe_float('d_A_c', total_c / 2.0),
        safe_float('d_B_c', total_c / 2.0),
        total_c, 0.1, 0.1
    )

    mtl = safe_float('M_TL', 0.0)
    mtr = safe_float('M_TR', 0.0)
    mtl, mtr = _scale_vertical_offsets_to_slider(mtl, mtr, wv)

    res = {k: safe_float(k) for k in FEATURE_ORDER}
    res.update({
        'H_s': hs_actual,
        'WWR_s': int(actual_wwr_s),
        'WWR_c': int(real_wwr_c),
        'W_s': ws_pane,
        'W_c': round(wc_actual, 2),
        'SH_s': shs_actual,
        'SH_c': round(shc_actual, 2),
        'd_A_s': d_a_s_actual,
        'd_B_s': d_b_s_actual,
        'd_A_c': d_a_c_actual,
        'd_B_c': d_b_c_actual,
        'N_v': nv,
        'M_TL': mtl,
        'M_TR': mtr,
        'L_oh': L_final,
        'd_mv': d_final
    })
    return res


BaseClassroomPredictorApp_V3 = ClassroomPredictorApp


class ClassroomPredictorApp(BaseClassroomPredictorApp_V3):
    def init_ui(self):
        super().init_ui()
        if 'W_v' in self.inputs:
            slider = self.inputs['W_v']
            _, _, val_lbl = self.rows['W_v']

            def sync_wv_label(v):
                if val_lbl is None:
                    return
                idx = max(0, min(int(v), len(_wv_slider_values()) - 1))
                val_lbl.setText(_format_feature_value('W_v', _wv_slider_values()[idx]))

            slider.valueChanged.connect(sync_wv_label)
            sync_wv_label(slider.value())

        for widget in self.inputs.values():
            if hasattr(widget, 'valueChanged'):
                widget.valueChanged.connect(lambda _value, self=self: self.update_viz_only(preserve_limits=True))

        self.cb_ori.currentTextChanged.connect(self.on_scheme_changed)
        self.cb_sha.currentTextChanged.connect(self.on_scheme_changed)
        self.reset_3d_view()

    def get_feature_value(self, feat):
        if feat == 'W_v':
            slider = self.inputs[feat]
            idx = max(0, min(slider.value(), len(_wv_slider_values()) - 1))
            return float(_wv_slider_values()[idx])
        return super().get_feature_value(feat)

    def set_feature_value(self, feat, real_val):
        if feat == 'W_v':
            slider = self.inputs[feat]
            idx = _nearest_wv_index(real_val)
            slider.blockSignals(True)
            slider.setValue(idx)
            slider.blockSignals(False)
            _, _, val_lbl = self.rows[feat]
            if val_lbl:
                val_lbl.setText(_format_feature_value(feat, _wv_slider_values()[idx]))
            return
        return super().set_feature_value(feat, real_val)

    def default_3d_view_state(self):
        return {
            'elev': 25,
            'azim': 45,
            'roll': 0.0,
        }

    def capture_3d_view_state(self):
        return {
            'elev': self.ax.elev,
            'azim': self.ax.azim,
            'roll': getattr(self.ax, 'roll', 0.0),
            'xlim': self.ax.get_xlim3d(),
            'ylim': self.ax.get_ylim3d(),
            'zlim': self.ax.get_zlim3d(),
        }

    def restore_3d_view_state(self, state, restore_limits=False):
        if not state:
            return
        try:
            self.ax.view_init(elev=state.get('elev', 25), azim=state.get('azim', 45), roll=state.get('roll', 0.0))
        except TypeError:
            self.ax.view_init(elev=state.get('elev', 25), azim=state.get('azim', 45))

        if restore_limits:
            self.ax.set_xlim3d(*state['xlim'])
            self.ax.set_ylim3d(*state['ylim'])
            self.ax.set_zlim3d(*state['zlim'])

    def on_scheme_changed(self, *args):
        self.reset_3d_view()

    def reset_3d_view(self, *args):
        sha_mode = self.cb_sha.currentText()
        raw_in = {f: self.get_feature_value(f) for f in FEATURE_ORDER}
        cal_data = geometry_engine(raw_in, sha_mode)
        draw_classroom_3d(self.ax, cal_data, sha_mode, self.chk_upper.isChecked(), self.chk_side.isChecked())
        self.restore_3d_view_state(self.default_3d_view_state())
        self.canvas.draw_idle()

    def update_viz_only(self, *args, preserve_limits=False):
        sha_mode = self.cb_sha.currentText()
        raw_in = {f: self.get_feature_value(f) for f in FEATURE_ORDER}
        cal_data = geometry_engine(raw_in, sha_mode)
        saved_view = self.capture_3d_view_state()
        draw_classroom_3d(self.ax, cal_data, sha_mode, self.chk_upper.isChecked(), self.chk_side.isChecked())
        self.restore_3d_view_state(saved_view, restore_limits=preserve_limits)
        self.canvas.draw_idle()

    def do_prediction(self):
        ori_mode = self.cb_ori.currentText()
        sha_mode = self.cb_sha.currentText()
        folder = self.model_map.get((ori_mode, sha_mode))
        if not folder:
            QMessageBox.warning(self, "模型缺失", "当前朝向与遮阳形式尚未找到对应模型目录。")
            return

        self.last_prediction = None
        self.btn_save_scheme.setEnabled(False)

        try:
            self.btn_run.setEnabled(False)
            QApplication.processEvents()

            raw_in = {f: self.get_feature_value(f) for f in FEATURE_ORDER}
            cal_data = geometry_engine(raw_in, sha_mode)

            if "悬挑" in sha_mode or "组合" in sha_mode:
                input_dmv = self.get_feature_value('d_mv')
                if input_dmv > cal_data['d_mv'] + 1e-9:
                    QMessageBox.warning(
                        self, "悬挑移动约束",
                        f"dmv 不能超过 0.8 + (3.8 - 0.8 - Hs - SHs)\n\n"
                        f"  计算后的 Hs = {cal_data['H_s']},  SHs = {cal_data['SH_s']}\n"
                        f"  最大允许值 = {round(max(0.0, 0.8 + (3.8 - 0.8 - cal_data['H_s'] - cal_data['SH_s'])), 2)}\n"
                        f"  您设定的 dmv = {input_dmv}\n\n"
                        f"dmv 已自动限制为 {cal_data['d_mv']}"
                    )

                alpha_val = self.get_feature_value('alpha_oh')
                if alpha_val < 0:
                    max_L_oh = round(cal_data['d_mv'] + cal_data['SH_s'], 2)
                    input_L_oh = self.get_feature_value('L_oh')
                    if input_L_oh > max_L_oh:
                        QMessageBox.warning(
                            self, "遮阳板长度约束",
                            f"当 αoh < 0 时，Loh 必须 ≤ dmv + SHs\n\n"
                            f"  当前 dmv = {cal_data['d_mv']},  SHs = {cal_data['SH_s']}\n"
                            f"  最大允许值 = {max_L_oh}\n"
                            f"  您设定的 Loh = {input_L_oh}\n\n"
                            f"Loh 已自动限制为 {cal_data['L_oh']}"
                        )

            if "垂直" in sha_mode or "组合" in sha_mode:
                input_mtl = raw_in.get('M_TL', 0.0)
                input_mtr = raw_in.get('M_TR', 0.0)
                wv_val = raw_in.get('W_v', 0.01)
                if input_mtl + input_mtr > wv_val:
                    QMessageBox.warning(
                        self, "垂直遮阳移动约束",
                        f"MTL + MTR 之和不能超过 Wv\n\n"
                        f"  当前 Wv = {_format_feature_value('W_v', wv_val)}\n"
                        f"  您设定的 MTL = {_format_feature_value('M_TL', input_mtl)},  MTR = {_format_feature_value('M_TR', input_mtr)}\n"
                        f"  合计 = {input_mtl + input_mtr:.2f} m  >  Wv = {_format_feature_value('W_v', wv_val)}\n\n"
                        f"已按比例自动缩放：MTL = {_format_feature_value('M_TL', cal_data['M_TL'])},  MTR = {_format_feature_value('M_TR', cal_data['M_TR'])}"
                    )

            writeback_keys = [
                'WWR_c', 'WWR_s', 'H_s', 'W_s', 'W_c',
                'SH_s', 'SH_c', 'd_A_s', 'd_B_s', 'd_A_c', 'd_B_c',
                'L_oh', 'd_mv', 'M_TL', 'M_TR'
            ]
            for key in writeback_keys:
                if key in cal_data:
                    self.set_feature_value(key, cal_data[key])

            active_feats = SHADE_FEATURE_MAP.get(sha_mode, FEATURE_ORDER[1:13])
            X = pd.DataFrame([[cal_data[f] for f in active_feats]], columns=active_feats)

            ckey = (ori_mode, sha_mode)
            if ckey not in self.model_cache:
                self.model_cache[ckey] = {
                    f"{t}_{m}": joblib.load(os.path.join(folder, f"{m}_model_{t}.joblib"))
                    for t in TARGET_KEYS for m in ["xgb", "lgbm", "rf", "meta"]
                }

            mc = self.model_cache[ckey]
            result_values = {}
            for t in TARGET_KEYS:
                m1 = mc[f"{t}_xgb"].predict(X)
                m2 = mc[f"{t}_lgbm"].predict(X)
                m3 = mc[f"{t}_rf"].predict(X)
                final = mc[f"{t}_meta"].predict(np.column_stack([m1, m2, m3]))[0]

                if t in ["sDA", "sGA"]:
                    final = max(0.0, min(final, 1.0))
                    v_show = final * 100
                elif t in ["UDI", "SVF"]:
                    v_show = max(0.0, min(final, 100.0))
                else:
                    v_show = final

                result_values[t] = float(v_show)
                self.res_labels[t].setText(f"{v_show:.2f}")

            saved_view = self.capture_3d_view_state()
            draw_classroom_3d(self.ax, cal_data, sha_mode, self.chk_upper.isChecked(), self.chk_side.isChecked())
            self.restore_3d_view_state(saved_view, restore_limits=True)
            self.canvas.draw_idle()

            self.last_prediction = {
                'ori': ori_mode,
                'shade': sha_mode,
                'raw_inputs': dict(raw_in),
                'calculated': dict(cal_data),
                'results': result_values
            }
            self.btn_save_scheme.setEnabled(True)
            self.btn_run.setEnabled(True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "运行故障", f"预测或计算出错：\n{str(e)}")
            self.btn_save_scheme.setEnabled(False)
            self.btn_run.setEnabled(True)


# %% [Cell 4H: 多视图覆盖 - 采光窗前视图 / 走廊窗后视图 / 透视图]

VIEW_PRESETS = {
    '采光窗侧 (前视图)': {'elev': 0, 'azim': 0, 'roll': 0.0, 'facade': 'daylight_front'},
    '走廊窗侧 (后视图)': {'elev': 0, 'azim': 180, 'roll': 0.0, 'facade': 'corridor_back'},
    '透视图':            {'elev': 25, 'azim': 45, 'roll': 0.0, 'facade': None},
}

VIEW_KEYS = list(VIEW_PRESETS.keys())


BaseClassroomPredictorApp_V4 = ClassroomPredictorApp


class ClassroomPredictorApp(BaseClassroomPredictorApp_V4):

    def init_ui(self):
        self.view_axes = {}
        self.view_canvases = {}
        self._view_container = None

        super().init_ui()

        self._orig_canvas = self.canvas
        self._orig_fig = self.fig

        self._right_layout = None
        main_layout = self.centralWidget().layout()
        for i in range(main_layout.count()):
            item = main_layout.itemAt(i)
            if item and item.layout():
                lay = item.layout()
                for j in range(lay.count()):
                    sub = lay.itemAt(j)
                    if sub and sub.widget() is self._orig_canvas:
                        self._right_layout = lay
                        self._canvas_layout_index = j
                        break
                if self._right_layout:
                    break

        grp_display = None
        for child in self.centralWidget().findChildren(QGroupBox):
            if child.title() == "3. 环境显示":
                grp_display = child
                break

        if grp_display:
            layout = grp_display.layout()

            sep = QLabel("── 多视图设置 ──")
            sep.setAlignment(Qt.AlignCenter)
            sep.setStyleSheet("color: #7f8c8d; font-size: 11px; padding-top: 6px;")
            layout.addWidget(sep)

            row_count = QHBoxLayout()
            row_count.addWidget(QLabel("视图窗口数:"))
            self.cb_view_count = QComboBox()
            self.cb_view_count.addItems(["1 个视图", "2 个视图", "3 个视图"])
            self.cb_view_count.setCurrentIndex(0)
            self.cb_view_count.currentIndexChanged.connect(self._on_view_layout_changed)
            row_count.addWidget(self.cb_view_count)
            layout.addLayout(row_count)

            self.view_checks = {}
            for i, vk in enumerate(VIEW_KEYS):
                chk = QCheckBox(vk)
                chk.setChecked(i == 2)
                chk.stateChanged.connect(self._on_view_selection_changed)
                self.view_checks[vk] = chk
                layout.addWidget(chk)

        self._rebuild_view_canvas()

    # ── 视图数量 / 选中状态变化 ──

    def _on_view_layout_changed(self, idx):
        count = idx + 1
        checked = [k for k in VIEW_KEYS if self.view_checks[k].isChecked()]

        if len(checked) > count:
            for k in reversed(checked[count:]):
                self.view_checks[k].blockSignals(True)
                self.view_checks[k].setChecked(False)
                self.view_checks[k].blockSignals(False)
        elif len(checked) < count:
            for k in VIEW_KEYS:
                if len(checked) >= count:
                    break
                if k not in checked:
                    self.view_checks[k].blockSignals(True)
                    self.view_checks[k].setChecked(True)
                    self.view_checks[k].blockSignals(False)
                    checked.append(k)

        self._rebuild_view_canvas()

    def _on_view_selection_changed(self, state):
        count = self.cb_view_count.currentIndex() + 1
        checked = [k for k in VIEW_KEYS if self.view_checks[k].isChecked()]

        if len(checked) > count:
            sender = self.sender()
            for k in VIEW_KEYS:
                if self.view_checks[k] is not sender and self.view_checks[k].isChecked():
                    if len(checked) <= count:
                        break
                    self.view_checks[k].blockSignals(True)
                    self.view_checks[k].setChecked(False)
                    self.view_checks[k].blockSignals(False)
                    checked.remove(k)

        if len(checked) == 0:
            for k in VIEW_KEYS:
                self.view_checks[k].blockSignals(True)
                self.view_checks[k].setChecked(True)
                self.view_checks[k].blockSignals(False)
                break

        self._rebuild_view_canvas()

    # ── 重建画布 ──

    def _active_view_keys(self):
        return [k for k in VIEW_KEYS if self.view_checks[k].isChecked()]

    def _rebuild_view_canvas(self):
        active = self._active_view_keys()
        if not active:
            active = [VIEW_KEYS[2]]

        old_figs_to_close = []
        for vk, (fig, canvas) in self.view_canvases.items():
            old_figs_to_close.append(fig)

        if self._view_container is not None:
            if self._right_layout:
                self._right_layout.removeWidget(self._view_container)
            self._view_container.setParent(None)
            self._view_container.deleteLater()
            self._view_container = None

        if self._orig_canvas is not None and self._orig_canvas.parent() is not None:
            if self._right_layout:
                self._right_layout.removeWidget(self._orig_canvas)
            self._orig_canvas.hide()
            self._orig_canvas.setParent(None)

        self._view_container = QWidget()
        n = len(active)
        if n == 1:
            container_layout = QVBoxLayout(self._view_container)
        else:
            container_layout = QHBoxLayout(self._view_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(4)

        self.view_axes = {}
        self.view_canvases = {}

        figw = max(4, 9 // n)
        figh = 7 if n <= 2 else 6

        for vk in active:
            panel = QVBoxLayout()
            title = QLabel(vk)
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet("font-size: 11px; font-weight: bold; color: #34495e; padding: 2px;")
            panel.addWidget(title)

            fig = plt.figure(figsize=(figw, figh))
            ax = fig.add_subplot(111, projection='3d')
            canvas = FigureCanvas(fig)
            panel.addWidget(canvas, 1)
            container_layout.addLayout(panel)

            self.view_axes[vk] = ax
            self.view_canvases[vk] = (fig, canvas)

        if VIEW_KEYS[2] in self.view_axes:
            self.ax = self.view_axes[VIEW_KEYS[2]]
            self.fig, self.canvas = self.view_canvases[VIEW_KEYS[2]]
        else:
            first_key = active[0]
            self.ax = self.view_axes[first_key]
            self.fig, self.canvas = self.view_canvases[first_key]

        if self._right_layout:
            self._right_layout.addWidget(self._view_container)
        else:
            self.centralWidget().layout().addWidget(self._view_container)

        for old_fig in old_figs_to_close:
            try:
                plt.close(old_fig)
            except Exception:
                pass

        self._redraw_all_views()

    # ── 绘制所有视图 ──

    def _redraw_all_views(self):
        sha_mode = self.cb_sha.currentText()
        raw_in = {f: self.get_feature_value(f) for f in FEATURE_ORDER}
        cal_data = geometry_engine(raw_in, sha_mode)
        show_upper = self.chk_upper.isChecked()
        show_side = self.chk_side.isChecked()

        for vk, ax in self.view_axes.items():
            preset = VIEW_PRESETS[vk]
            draw_classroom_3d(ax, cal_data, sha_mode, show_upper, show_side, facade_filter=preset.get('facade'))
            try:
                ax.view_init(elev=preset['elev'], azim=preset['azim'], roll=preset.get('roll', 0.0))
            except TypeError:
                ax.view_init(elev=preset['elev'], azim=preset['azim'])

        for vk, (fig, canvas) in self.view_canvases.items():
            canvas.draw_idle()

    # ── 覆盖视角恢复/重置 ──

    def default_3d_view_state(self):
        return VIEW_PRESETS[VIEW_KEYS[2]].copy()

    def reset_3d_view(self, *args):
        self._redraw_all_views()

    def update_viz_only(self, *args, preserve_limits=False):
        sha_mode = self.cb_sha.currentText()
        raw_in = {f: self.get_feature_value(f) for f in FEATURE_ORDER}
        cal_data = geometry_engine(raw_in, sha_mode)
        show_upper = self.chk_upper.isChecked()
        show_side = self.chk_side.isChecked()

        persp_key = VIEW_KEYS[2]
        saved_view = None
        if persp_key in self.view_axes:
            saved_view = self.capture_3d_view_state()

        for vk, ax in self.view_axes.items():
            preset = VIEW_PRESETS[vk]
            draw_classroom_3d(ax, cal_data, sha_mode, show_upper, show_side, facade_filter=preset.get('facade'))
            if vk == persp_key and saved_view:
                self.restore_3d_view_state(saved_view, restore_limits=preserve_limits)
            else:
                try:
                    ax.view_init(elev=preset['elev'], azim=preset['azim'], roll=preset.get('roll', 0.0))
                except TypeError:
                    ax.view_init(elev=preset['elev'], azim=preset['azim'])

        for vk, (fig, canvas) in self.view_canvases.items():
            canvas.draw_idle()

    def do_prediction(self):
        ori_mode = self.cb_ori.currentText()
        sha_mode = self.cb_sha.currentText()
        folder = self.model_map.get((ori_mode, sha_mode))
        if not folder:
            QMessageBox.warning(self, "模型缺失", "当前朝向与遮阳形式尚未找到对应模型目录。")
            return

        self.last_prediction = None
        self.btn_save_scheme.setEnabled(False)

        try:
            self.btn_run.setEnabled(False)
            QApplication.processEvents()

            raw_in = {f: self.get_feature_value(f) for f in FEATURE_ORDER}
            cal_data = geometry_engine(raw_in, sha_mode)

            if "悬挑" in sha_mode or "组合" in sha_mode:
                input_dmv = self.get_feature_value('d_mv')
                if input_dmv > cal_data['d_mv'] + 1e-9:
                    QMessageBox.warning(
                        self, "悬挑移动约束",
                        f"dmv 不能超过 0.8 + (3.8 - 0.8 - Hs - SHs)\n\n"
                        f"  计算后的 Hs = {cal_data['H_s']},  SHs = {cal_data['SH_s']}\n"
                        f"  最大允许值 = {round(max(0.0, 0.8 + (3.8 - 0.8 - cal_data['H_s'] - cal_data['SH_s'])), 2)}\n"
                        f"  您设定的 dmv = {input_dmv}\n\n"
                        f"dmv 已自动限制为 {cal_data['d_mv']}"
                    )

                alpha_val = self.get_feature_value('alpha_oh')
                if alpha_val < 0:
                    max_L_oh = round(cal_data['d_mv'] + cal_data['SH_s'], 2)
                    input_L_oh = self.get_feature_value('L_oh')
                    if input_L_oh > max_L_oh:
                        QMessageBox.warning(
                            self, "遮阳板长度约束",
                            f"当 αoh < 0 时，Loh 必须 ≤ dmv + SHs\n\n"
                            f"  当前 dmv = {cal_data['d_mv']},  SHs = {cal_data['SH_s']}\n"
                            f"  最大允许值 = {max_L_oh}\n"
                            f"  您设定的 Loh = {input_L_oh}\n\n"
                            f"Loh 已自动限制为 {cal_data['L_oh']}"
                        )

            if "垂直" in sha_mode or "组合" in sha_mode:
                input_mtl = raw_in.get('M_TL', 0.0)
                input_mtr = raw_in.get('M_TR', 0.0)
                wv_val = raw_in.get('W_v', 0.01)
                if input_mtl + input_mtr > wv_val:
                    QMessageBox.warning(
                        self, "垂直遮阳移动约束",
                        f"MTL + MTR 之和不能超过 Wv\n\n"
                        f"  当前 Wv = {_format_feature_value('W_v', wv_val)}\n"
                        f"  您设定的 MTL = {_format_feature_value('M_TL', input_mtl)},  MTR = {_format_feature_value('M_TR', input_mtr)}\n"
                        f"  合计 = {input_mtl + input_mtr:.2f} m  >  Wv = {_format_feature_value('W_v', wv_val)}\n\n"
                        f"已按比例自动缩放：MTL = {_format_feature_value('M_TL', cal_data['M_TL'])},  MTR = {_format_feature_value('M_TR', cal_data['M_TR'])}"
                    )

            writeback_keys = [
                'WWR_c', 'WWR_s', 'H_s', 'W_s', 'W_c',
                'SH_s', 'SH_c', 'd_A_s', 'd_B_s', 'd_A_c', 'd_B_c',
                'L_oh', 'd_mv', 'M_TL', 'M_TR'
            ]
            for key in writeback_keys:
                if key in cal_data:
                    self.set_feature_value(key, cal_data[key])

            ckey = (ori_mode, sha_mode)
            if ckey not in self.model_cache:
                cache = {}
                for t in TARGET_KEYS:
                    for m in ["xgb", "lgbm", "rf", "meta"]:
                        cache[f"{t}_{m}"] = joblib.load(os.path.join(folder, f"{m}_model_{t}.joblib"))
                xtrain_path = os.path.join(folder, "X_train.joblib")
                if os.path.exists(xtrain_path):
                    cache['_feature_order'] = joblib.load(xtrain_path).columns.tolist()
                else:
                    cache['_feature_order'] = None
                self.model_cache[ckey] = cache

            mc = self.model_cache[ckey]
            model_feats = mc.get('_feature_order')
            if model_feats:
                row = [cal_data.get(FEATURE_ALIAS.get(f, f), 0.0) for f in model_feats]
                X = pd.DataFrame([row], columns=model_feats)
            else:
                active_feats = SHADE_FEATURE_MAP.get(sha_mode, FEATURE_ORDER[1:13])
                X = pd.DataFrame([[cal_data[f] for f in active_feats]], columns=active_feats)
            result_values = {}
            for t in TARGET_KEYS:
                m1 = mc[f"{t}_xgb"].predict(X)
                m2 = mc[f"{t}_lgbm"].predict(X)
                m3 = mc[f"{t}_rf"].predict(X)
                final = mc[f"{t}_meta"].predict(np.column_stack([m1, m2, m3]))[0]

                if t in ["sDA", "sGA"]:
                    final = max(0.0, min(final, 1.0))
                    v_show = final * 100
                elif t in ["UDI", "SVF"]:
                    v_show = max(0.0, min(final, 100.0))
                else:
                    v_show = final

                result_values[t] = float(v_show)
                self.res_labels[t].setText(f"{v_show:.2f}")

            persp_key = VIEW_KEYS[2]
            saved_view = None
            if persp_key in self.view_axes:
                saved_view = self.capture_3d_view_state()

            show_upper = self.chk_upper.isChecked()
            show_side = self.chk_side.isChecked()
            for vk, ax in self.view_axes.items():
                preset = VIEW_PRESETS[vk]
                draw_classroom_3d(ax, cal_data, sha_mode, show_upper, show_side, facade_filter=preset.get('facade'))
                if vk == persp_key and saved_view:
                    self.restore_3d_view_state(saved_view, restore_limits=True)
                else:
                    try:
                        ax.view_init(elev=preset['elev'], azim=preset['azim'], roll=preset.get('roll', 0.0))
                    except TypeError:
                        ax.view_init(elev=preset['elev'], azim=preset['azim'])

            for vk, (fig, canvas) in self.view_canvases.items():
                canvas.draw_idle()

            self.last_prediction = {
                'ori': ori_mode,
                'shade': sha_mode,
                'raw_inputs': dict(raw_in),
                'calculated': dict(cal_data),
                'results': result_values
            }
            self.btn_save_scheme.setEnabled(True)
            self.btn_run.setEnabled(True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "运行故障", f"预测或计算出错：\n{str(e)}")
            self.btn_save_scheme.setEnabled(False)
            self.btn_run.setEnabled(True)


# %% [Cell 5: 程序启动入口]
if __name__ == "__main__":
    app = QApplication.instance() or QApplication(sys.argv)
    main_win = ClassroomPredictorApp()
    main_win.show()
    app.exec()
