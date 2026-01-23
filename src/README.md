
# manga-dev：整体处理流程说明

## 项目概要

### 数据源文件

* drpall
* MaNGA Firefly Stellar Populations
* manga-MAPS-DAPTYPE

### 星系筛选(倾斜角)

#### 数据

* drpall -- NSA_SERSIC_BA

#### 处理

* 根据NSA_SERSIC_BA的值,近似计算星系倾斜角i。
* 筛选i在[25, 70]之间的星系，共8070个。

### 星系旋转曲线拟合

#### 公式

* （1） Vobs = Vsys + Vrot *(sin(i)* cos(phi - phi_0))
* （2） Vrot(r) = Vc *tanh(r / Rt) + s_out* r

#### 数据来源

* MAPS -- ECOOPA
* MAPS -- ECOOELL
* MAPS -- EMLINE_GVEL
* MAPS -- EMLINE_GVEL_MASK
* MAPS -- EMLINE_GVEL_IVAR

#### 旋转曲线处理

* 提取MPAS文件EMLINE_GVEL，获取星系`Vobs`
* 筛选S/N > 10，并且phi < 45◦ 的数据
* 根据公式（1）（2），使用优化方法拟合，拟合参数 Vc, Rt, s_out, Vsys, inc, phi_delta
* 根据residual，过滤掉部分`Vobs`数据
* 根据拟合的reduced chi-square和NRMSE值，筛选星系

### 暗物质（NFW）推断

#### 公式

* （4） Vrot^2 = Vstar^2 + Vdm^2 - Vdrift^2
* （5） Vdm(r)^2 = ((10 *G* H(z) *M200)^2 / x)* (ln(1 + c*x) - (c*x)/(1 + c*x)) / (ln(1 + c) - c/(1 + c))
* （6） Vstar(r)^2 = (G *MB* r) / (r + a)^2 +(2 *G* M_baryon / Rd) *y^2* [I_0(y) K_0(y) - I_1(y) K_1(y)]
* （7） Vdrift(r)^2 = 2 *sigma_0^2* (r / R_d)

#### 数据

* 观测速度`Vobs`：经过旋转曲线拟合过程过滤

#### 处理

构建 PyMC 模型并做贝叶斯推断

* prior
  * M200: 公式(5)的参数
  * c: 公式(5)的参数
  * Mstar: 公式(6)的参数
  * Re: 公式(6)(1)的参数
  * f_bulge: 公式(6)的参数
  * a: 公式(6)的参数
  * sigma_0: 公式(7)的参数
  * v_sys: 公式(1)的参数
* deterministic relations
  * 根据公式(5),(6),(7)分别计算Vstar^2, Vdm^2, Vdrift^2
  * 根据公式(4)计算`Vrot`
  * 根据公式(1)计算`Vobs_model`
* likelihood
  * mu: `Vobs_model`
  * observed: `Vobs`
