
# manga-dev：整体处理流程说明

## 项目概要

### 数据源文件
* drpall
* MaNGA Firefly Stellar Populations
* manga-MAPS-DAPTYPE

### 星系筛选(倾斜角)
1. 数据
* drpall -- NSA_SERSIC_BA

2. 处理
* 根据NSA_SERSIC_BA的值，近似计算星系倾斜角i。
* 筛选i在[25, 70]之间的星系，共8070个。

### 星系旋转曲线拟合
1. 公式
* （1） Vobs = Vsys + Vrot * (sin(i) * cos(phi - phi_0))
* （2） Vrot(r) = Vc * tanh(r / Rt) + s_out * r

2. 数据
* MAPS -- ECOOPA
* MAPS -- ECOOELL
* MAPS -- EMLINE_GVEL
* MAPS -- EMLINE_GVEL_MASK
* MAPS -- EMLINE_GVEL_IVAR

3. 处理
* 提取MPAS文件EMLINE_GVEL，获取星系Vobs
* 筛选S/N > 10，并且phi < 45◦ 的数据
* 根据公式（1）（2），使用优化方法拟合，拟合参数 Vc, Rt, s_out, Vsys, inc, phi_delta
* 根据residual，过滤掉部分Vobs数据
* 根据拟合的reduced chi-square和NRMSE值，筛选星系

### 恒星质量拟合
1. 公式
* （3） Mstar(r) = MB * r^2 / (r + a)^2 + MD * (1 - (1 + r / rd) * exp(-r / rd))

2. 数据
* Firefly -- HDU11: STELLAR MASS
* drpall -- NSA_ELPETRO_MASS

. 处理
* 提取Firefly文件数据STELLAR MASS，获取Star mass cell
* 提取drpall文件NSA_ELPETRO_MASS，获取Star total mass
* 使用优化方法拟合，拟合参数Re

### 暗物质（NFW）推断
1. 公式
* （4） Vrot^2 = Vstar^2 + Vdm^2 - Vdrift^2
* （5） Vdm^2 = ((10 * G * H(z) * M200)^2 / x) * (ln(1 + c*x) - (c*x)/(1 + c*x)) / (ln(1 + c) - c/(1 + c))
* （6） Vstar^2 = (G * MB * r) / (r + a)^2 +(2 * G * M_baryon / Rd) * y^2 * [I_0(y) K_0(y) - I_1(y) K_1(y)]
* （7）  Vdrift^2 = 2 * sigma_0^2 * (R / R_d)

2. 数据
* 观测速度Vobs：经过旋转曲线拟合过程过滤
* Re：由恒星质量拟合过程返回)
* Mstar: drpall -- NSA_ELPETRO_MASS

3. 处理
* 构建 PyMC 模型并做贝叶斯推断
* 先验：`M200`, `c`, `sigma_0`, `R_d`, `v_sys`, `inc`, `f_bulge`, `a`
* 确定性关系
  * 根据公式(5),(6),(7)分别计算Vstar^2, Vdm^2, Vdrift^2
  * 根据公式(4)计算`Vrot`
  * 根据公式(1)计算`Vobs_model`
  * likelihood, observed值为`Vobs`
