import taichi as ti
import taichi.math as tm

# 初始化 Taichi
ti.init(arch=ti.vulkan)

# 场景参数
image_width = 800
image_height = 600

# 摄像机位置
camera_pos = tm.vec3(0.0, 0.0, 8.0)

# 光源位置
light_pos = tm.vec3(2.0, 3.0, 4.0)
light_color = tm.vec3(1.0, 1.0, 1.0)

# 几何体参数
# 红色球体
sphere_center = tm.vec3(-1.2, -0.2, 0.0)
sphere_radius = 1.2
sphere_color = tm.vec3(0.8, 0.1, 0.1)

# 紫色圆锥
cone_apex = tm.vec3(1.2, 1.2, 0.0)
cone_base_y = -1.4
cone_base_radius = 1.2
cone_color = tm.vec3(0.6, 0.2, 0.8)

# 背景颜色
background_color = tm.vec3(0.0, 0.2, 0.3)

# 图像缓冲区
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(image_width, image_height))

# UI 可调控参数（使用 ti.field 以便在 kernel 中访问）
Ka = ti.field(dtype=ti.f32, shape=())  # 环境光系数
Kd = ti.field(dtype=ti.f32, shape=())  # 漫反射系数
Ks = ti.field(dtype=ti.f32, shape=())  # 镜面高光系数
Shininess = ti.field(dtype=ti.f32, shape=())  # 高光指数

# 切换开关
use_blinn_phong = ti.field(dtype=ti.i32, shape=())  # 是否使用 Blinn-Phong
enable_shadow = ti.field(dtype=ti.i32, shape=())  # 是否启用阴影

# 初始化默认值
@ti.kernel
def init_params():
    Ka[None] = 0.2
    Kd[None] = 0.7
    Ks[None] = 0.5
    Shininess[None] = 32.0
    use_blinn_phong[None] = 0
    enable_shadow[None] = 0


@ti.func
def intersect_sphere(ray_origin: tm.vec3, ray_dir: tm.vec3) -> tm.vec2:
    """
    计算射线与球体的交点
    返回: tm.vec2(t1, t2)，如果没有交点返回 tm.vec2(-1.0, -1.0)
    """
    oc = ray_origin - sphere_center
    a = tm.dot(ray_dir, ray_dir)
    b = 2.0 * tm.dot(oc, ray_dir)
    c = tm.dot(oc, oc) - sphere_radius * sphere_radius
    discriminant = b * b - 4.0 * a * c

    t_min = -1.0
    t_max = -1.0

    if discriminant >= 0.0:
        sqrt_disc = tm.sqrt(discriminant)
        t_min = (-b - sqrt_disc) / (2.0 * a)
        t_max = (-b + sqrt_disc) / (2.0 * a)

    return tm.vec2(t_min, t_max)


@ti.func
def intersect_cone(ray_origin: tm.vec3, ray_dir: tm.vec3) -> tm.vec2:
    """
    计算射线与圆锥的交点
    圆锥顶点在 cone_apex，底面在 y = cone_base_y，底面半径为 cone_base_radius
    返回: tm.vec2(t1, t2)，如果没有交点返回 tm.vec2(-1.0, -1.0)
    """
    # 圆锥参数
    apex = cone_apex
    h = cone_base_y - apex.y  # 圆锥高度（负值，因为底面在顶点下方）
    r = cone_base_radius

    # 计算圆锥的半角度
    # 圆锥的侧面可以表示为: (x - apex.x)^2 + (z - apex.z)^2 = (r/h)^2 * (y - apex.y)^2

    # 将射线转换到以顶点为原点的坐标系
    oc = ray_origin - apex

    # 圆锥侧面方程的二次项系数
    # 设射线为 P = O + tD
    # 需要解: (Ox + t*Dx)^2 + (Oz + t*Dz)^2 = (r/h)^2 * (Oy + t*Dy)^2

    k = r / h  # 斜率比例
    k2 = k * k

    a = ray_dir.x * ray_dir.x + ray_dir.z * ray_dir.z - k2 * ray_dir.y * ray_dir.y
    b = 2.0 * (oc.x * ray_dir.x + oc.z * ray_dir.z - k2 * oc.y * ray_dir.y)
    c = oc.x * oc.x + oc.z * oc.z - k2 * oc.y * oc.y

    t_min = -1.0
    t_max = -1.0

    # 处理 a 接近 0 的情况（射线平行于圆锥母线）
    if ti.abs(a) < 1e-6:
        if ti.abs(b) > 1e-6:
            t_min = -c / b
            # 检查是否在有效高度范围内
            hit_point = ray_origin + t_min * ray_dir
            if hit_point.y <= apex.y and hit_point.y >= cone_base_y:
                t_max = t_min
    else:
        discriminant = b * b - 4.0 * a * c
        if discriminant >= 0.0:
            sqrt_disc = tm.sqrt(discriminant)
            t1 = (-b - sqrt_disc) / (2.0 * a)
            t2 = (-b + sqrt_disc) / (2.0 * a)

            # 检查交点是否在圆锥的有效部分（顶点到底面之间）
            hit1 = ray_origin + t1 * ray_dir
            hit2 = ray_origin + t2 * ray_dir

            valid1 = hit1.y <= apex.y and hit1.y >= cone_base_y
            valid2 = hit2.y <= apex.y and hit2.y >= cone_base_y

            # 还需要检查射线是否从内部穿过
            # 简化的处理方式：选择有效的、最小的正 t 值
            if t1 > 0 and valid1:
                t_min = t1
                if t2 > 0 and valid2:
                    t_max = t2
            elif t2 > 0 and valid2:
                t_min = t2

    # 检查底面圆盘
    # 底面平面: y = cone_base_y
    if ti.abs(ray_dir.y) > 1e-6:
        t_base = (cone_base_y - ray_origin.y) / ray_dir.y
        if t_base > 0:
            hit_base = ray_origin + t_base * ray_dir
            dist_from_center = tm.sqrt((hit_base.x - apex.x) ** 2 + (hit_base.z - apex.z) ** 2)
            if dist_from_center <= r:
                # 底面是有效的交点
                if t_min < 0 or (t_base < t_min and t_base > 0):
                    t_max = t_min if t_min > 0 else t_base
                    t_min = t_base
                elif t_max < 0 or (t_base < t_max and t_base > t_min):
                    t_max = t_base

    return tm.vec2(t_min, t_max)


@ti.func
def get_sphere_normal(hit_point: tm.vec3) -> tm.vec3:
    """获取球体表面法向量"""
    return (hit_point - sphere_center).normalized()


@ti.func
def get_cone_normal(hit_point: tm.vec3) -> tm.vec3:
    """获取圆锥表面法向量"""
    apex = cone_apex
    h = cone_base_y - apex.y
    r = cone_base_radius

    # 判断是侧面还是底面
    result = tm.vec3(0.0, -1.0, 0.0)
    if ti.abs(hit_point.y - cone_base_y) >= 1e-3:
        # 侧面法向量
        # 圆锥侧面的法向量方向
        k = r / h
        # 从顶点指向交点的水平向量
        dx = hit_point.x - apex.x
        dz = hit_point.z - apex.z
        # 侧面法向量
        ny = r / tm.sqrt(r * r + h * h)
        # 归一化
        horizontal_dist = tm.sqrt(dx * dx + dz * dz)
        if horizontal_dist > 1e-6:
            nx = dx / horizontal_dist * ti.abs(h) / tm.sqrt(r * r + h * h)
            nz = dz / horizontal_dist * ti.abs(h) / tm.sqrt(r * r + h * h)
            result = tm.vec3(nx, ny, nz).normalized()
        else:
            result = tm.vec3(0.0, 1.0, 0.0)
    return result


@ti.func
def phong_shading(hit_point: tm.vec3, normal: tm.vec3, view_dir: tm.vec3,
                  object_color: tm.vec3, in_shadow: ti.i32) -> tm.vec3:
    """
    Phong 光照模型计算
    """
    # 光源方向（指向光源）
    light_dir = (light_pos - hit_point).normalized()

    # 环境光
    ambient = Ka[None] * light_color * object_color

    # 漫反射
    diffuse = tm.vec3(0.0, 0.0, 0.0)
    # 镜面高光
    specular = tm.vec3(0.0, 0.0, 0.0)

    if not in_shadow:
        N_dot_L = tm.dot(normal, light_dir)
        N_dot_L = ti.max(0.0, N_dot_L)
        diffuse = Kd[None] * N_dot_L * light_color * object_color

        if use_blinn_phong[None] == 1:
            # Blinn-Phong 模型：使用半程向量
            half_dir = (light_dir + view_dir).normalized()
            N_dot_H = tm.dot(normal, half_dir)
            N_dot_H = ti.max(0.0, N_dot_H)
            spec = tm.pow(N_dot_H, Shininess[None])
            specular = Ks[None] * spec * light_color
        else:
            # Phong 模型：使用反射向量
            R = (2.0 * tm.dot(normal, light_dir) * normal - light_dir).normalized()
            R_dot_V = tm.dot(R, view_dir)
            R_dot_V = ti.max(0.0, R_dot_V)
            spec = tm.pow(R_dot_V, Shininess[None])
            specular = Ks[None] * spec * light_color

    # 合并所有分量
    color = ambient + diffuse + specular

    # 限制颜色范围
    color = tm.clamp(color, 0.0, 1.0)

    return color


@ti.func
def check_shadow(hit_point: tm.vec3) -> ti.i32:
    """
    检查点是否在阴影中
    向光源发射阴影射线，如果击中任何物体则返回 1（在阴影中）
    """
    # 使用变量保存结果，避免在非静态条件中 return
    in_shadow = 0

    if enable_shadow[None] == 1:
        # 阴影射线方向（指向光源）
        shadow_dir = (light_pos - hit_point).normalized()
        shadow_origin = hit_point + shadow_dir * 1e-4  # 避免自相交

        # 到光源的距离
        dist_to_light = tm.length(light_pos - hit_point)

        # 检查与球体的交点
        sphere_hit = intersect_sphere(shadow_origin, shadow_dir)
        hit_sphere = sphere_hit.x > 0 and sphere_hit.x < dist_to_light

        # 检查与圆锥的交点
        cone_hit = intersect_cone(shadow_origin, shadow_dir)
        hit_cone = cone_hit.x > 0 and cone_hit.x < dist_to_light

        # 如果击中任何物体，则在阴影中
        if hit_sphere or hit_cone:
            in_shadow = 1

    return in_shadow


@ti.kernel
def render():
    """渲染场景的主 kernel"""
    for i, j in pixels:
        # 将像素坐标归一化到 [-1, 1] 范围
        u = (i / image_width) * 2.0 - 1.0
        v = (j / image_height) * 2.0 - 1.0

        # 考虑屏幕宽高比
        aspect_ratio = image_width / image_height
        u = u * aspect_ratio

        # 创建射线
        ray_origin = camera_pos
        # 射线方向：从摄像机指向屏幕上的点 (u, v, 0)
        ray_dir = tm.vec3(u, v, -3.5).normalized()

        # 初始化颜色为背景色
        final_color = background_color
        closest_t = -1.0

        # 计算与球体的交点
        sphere_hit = intersect_sphere(ray_origin, ray_dir)
        sphere_t = sphere_hit.x if sphere_hit.x > 0 else sphere_hit.y

        # 计算与圆锥的交点
        cone_hit = intersect_cone(ray_origin, ray_dir)
        cone_t = cone_hit.x if cone_hit.x > 0 else cone_hit.y

        # 深度测试：选择最近的交点
        hit_object = 0  # 0 = 无, 1 = 球体, 2 = 圆锥

        if sphere_t > 0 and cone_t > 0:
            if sphere_t < cone_t:
                closest_t = sphere_t
                hit_object = 1
            else:
                closest_t = cone_t
                hit_object = 2
        elif sphere_t > 0:
            closest_t = sphere_t
            hit_object = 1
        elif cone_t > 0:
            closest_t = cone_t
            hit_object = 2

        # 着色
        if hit_object > 0:
            hit_point = ray_origin + closest_t * ray_dir
            view_dir = (camera_pos - hit_point).normalized()

            if hit_object == 1:
                # 球体
                normal = get_sphere_normal(hit_point)
                in_shadow = check_shadow(hit_point)
                final_color = phong_shading(hit_point, normal, view_dir, sphere_color, in_shadow)
            else:
                # 圆锥
                normal = get_cone_normal(hit_point)
                in_shadow = check_shadow(hit_point)
                final_color = phong_shading(hit_point, normal, view_dir, cone_color, in_shadow)

        pixels[i, j] = final_color


def main():
    # 初始化参数
    init_params()

    # 创建窗口
    window = ti.ui.Window("Phong Shading Renderer", (image_width, image_height))
    canvas = window.get_canvas()

    while window.running:
        # 处理事件
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                window.running = False

        # 渲染场景
        render()

        # 设置画布
        canvas.set_image(pixels)

        # 创建 GUI 控制面板
        window.GUI.begin("Phong Shading Controls", 0.02, 0.02, 0.25, 0.25)

        # Ka 滑动条
        old_ka = Ka[None]
        new_ka = window.GUI.slider_float("Ka (Ambient)", old_ka, 0.0, 1.0)
        if new_ka != old_ka:
            Ka[None] = new_ka

        # Kd 滑动条
        old_kd = Kd[None]
        new_kd = window.GUI.slider_float("Kd (Diffuse)", old_kd, 0.0, 1.0)
        if new_kd != old_kd:
            Kd[None] = new_kd

        # Ks 滑动条
        old_ks = Ks[None]
        new_ks = window.GUI.slider_float("Ks (Specular)", old_ks, 0.0, 1.0)
        if new_ks != old_ks:
            Ks[None] = new_ks

        # Shininess 滑动条
        old_shininess = Shininess[None]
        new_shininess = window.GUI.slider_float("Shininess", old_shininess, 1.0, 128.0)
        if new_shininess != old_shininess:
            Shininess[None] = new_shininess

        window.GUI.end()

        # 附加功能控制面板
        window.GUI.begin("Advanced Features", 0.02, 0.30, 0.25, 0.15)

        # Blinn-Phong 切换
        old_blinn = use_blinn_phong[None]
        new_blinn = window.GUI.checkbox("Use Blinn-Phong", old_blinn == 1)
        use_blinn_phong[None] = 1 if new_blinn else 0

        # 阴影切换
        old_shadow = enable_shadow[None]
        new_shadow = window.GUI.checkbox("Enable Shadow", old_shadow == 1)
        enable_shadow[None] = 1 if new_shadow else 0

        window.GUI.end()

        # 显示帧率信息
        window.GUI.begin("Info", 0.02, 0.48, 0.15, 0.06)
        window.GUI.text("ESC to exit")
        window.GUI.end()

        # 更新窗口
        window.show()


if __name__ == "__main__":
    main()
