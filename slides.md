---
# try also 'default' to start simple
theme: default
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
background: https://source.unsplash.com/collection/94734566/1920x1080
# apply any windi css classes to the current slide
class: 'text-center'
# https://sli.dev/custom/highlighters.html
highlighter: shiki
# show line numbers in code blocks
lineNumbers: false
# some information about the slides, markdown enabled
info: |
  ## Slidev Starter Template
  Presentation slides for developers.

  Learn more at [Sli.dev](https://sli.dev)
# persist drawings in exports and build
drawings:
  persist: false
---

# 整数线性规划问题（ILP）及其在深度学习编译器中的运用

汇报人： 罗翔 指导老师： 尚笠教授

<div class="abs-br m-6 flex gap-2">
  <button @click="$slidev.nav.openInEditor()" title="Open in Editor" class="text-xl icon-btn opacity-50 !border-none !hover:text-white">
    <carbon:edit />
  </button>
  <a href="https://github.com/slidevjs/slidev" target="_blank" alt="GitHub"
    class="text-xl icon-btn opacity-50 !border-none !hover:text-white">
    <carbon-logo-github />
  </a>
</div>

<!--
The last comment block of each slide will be treated as slide notes. It will be visible and editable in Presenter Mode along with the slide. [Read more in the docs](https://sli.dev/guide/syntax.html#notes)
-->

---

<br>

<style>
  ul {
    font-size: 30px;
  }

  li {
    font-size: 26px;
    margin: 15px 0;
  }

  li.transparent {
    color: #9ea7b3de
  }
</style>

<ul>
  整数线性规划问题（Integer linear programming）
  <li>单纯形算法（Simplex method）</li>


  <li class="transparent">整数单纯形算法（Simplex method + Gomory cut）</li>
  
  <li class="transparent">字典序最小问题（Lexicographical minimum）</li>

  <li class="transparent">整数字典序最小问题（Lexicographical minimum + Gomory cut）</li>
</ul>

---

<br>

<style>
  ul {
    font-size: 30px;
  }

  li {
    font-size: 26px;
    margin: 15px 0;
  }

  li.transparent {
    color: #9ea7b3de
  }
</style>

<ul>
  整数线性规划问题（Integer linear programming）
  <li class="transparent">单纯形算法（Simplex method）</li>


  <li>整数单纯形算法（Simplex method + Gomory cut）</li>
  
  <li class="transparent">字典序最小问题（Lexicographical minimum）</li>

  <li class="transparent">整数字典序最小问题（Lexicographical minimum + Gomory cut）</li>
</ul>

---

<br>

<style>
  ul {
    font-size: 30px;
  }

  li {
    font-size: 26px;
    margin: 15px 0;
  }

  li.transparent {
    color: #9ea7b3de
  }
</style>

<ul>
  整数线性规划问题（Integer linear programming）
  <li class="transparent">单纯形算法（Simplex method）</li>


  <li class="transparent">整数单纯形算法（Simplex method + Gomory cut）</li>
  
  <li>字典序最小问题（Lexicographical minimum）</li>

  <li class="transparent">整数字典序最小问题（Lexicographical minimum + Gomory cut）</li>
</ul>

---
layout: two-cols
---

# 问题定义

<br>

$R^n$ 上的字典序最小解：给定 m $\times$ n 的矩阵 M，m 维的向量 $\bold{v}$

令 $\bold{F} = \{ \bold{x} | \bold{x} \ge \bold{0}, M\bold{x} + \bold{v} \ge \bold{0}, \bold{x} \in R^n \}$

集合 $\bold{F}$ 为问题的可行域，判定集合 $\bold{F}$ 是否为空，若不为空则求出集合中字典序最小的元素。

::right::
<img src="/feasible.png" class="m-20 h-65" />

$\bold{F} = \{ (x_1, x_2, x_3) | x_i \ge 0, x_1 + x_2 + x_3 = 1 \}$

<br>

---

<br>

原可行域：

$$\bold{F} = \{ \bold{x} | \bold{x} \ge \bold{0}, M\bold{x} + \bold{v} \ge \bold{0}\}$$

对 $\bold{x}$ 作线性变换，引入 n $\times$ n 的矩阵 P，n 维的向量 $\bold{u}$

$$\bold{x} = P\bold{y} + \bold{u}$$

新可行域：

$$
\begin{array}{c}
\bold{F^*} & = \{ P\bold{y} + \bold{u} | P\bold{y} + \bold{u} \ge \bold{0}, M(P\bold{y} + \bold{u}) + \bold{v} \ge \bold{0}\} \\

& = \{ P\bold{y} + \bold{u} | P\bold{y} + \bold{u} \ge \bold{0}, MP\bold{y} + (M \bold{u} + \bold{v}) \ge \bold{0}\}
\end{array}
$$

<v-click>

对上述形式进一步抽象

$$
\bold{F} = \{ A\bold{y} + \bold{b} | \bold{x} = A\bold{y} + \bold{b} \ge \bold{0}, \bold{z} = C\bold{y} + \bold{d} \ge \bold{0}, \red{\bold{y} \ge \bold{0}} \}
$$

上述形式同样可以表示原可行域： A 是 n 阶单位阵($I_n$)，$\bold{b}$ 是零向量($\bold{0}$)，C 是 M，$\bold{d}$ 是 $\bold{v}$

$$
\bold{F} = \{ \bold{y} | \bold{x} = \bold{y} \ge \bold{0}, \bold{z} = M\bold{y} + \bold{v} \ge \bold{0}, \red{\bold{y} \ge \bold{0}} \}
$$

</v-click>

---

<br>

现有如下通用的可行域表示形式

$$
\bold{F} = \{ A\bold{y} + \bold{b} | \bold{x} = A\bold{y} + \bold{b} \ge \bold{0}, \bold{z} = C\bold{y} + \bold{d} \ge \bold{0}, \bold{y} \ge \bold{0} \}
$$

将其表示为矩阵形式

$$
\left[ \begin{array}{c} A \\ C \end{array} \right] \bold{y} + 
\left[ \begin{array}{c} \bold{b} \\ \bold{d} \end{array} \right]
= \left[ \begin{array}{c} \bold{x} \\ \bold{z} \end{array} \right]
$$

我们将 A 和 C 看作 (n + m) $\times$ n 矩阵 $S = \left[ \begin{array}{c} A \\ C \end{array} \right]$ 的分块矩阵，将 $\left[ \begin{array}{c} \bold{b} \\ \bold{d} \end{array} \right]$ 看作是 (m + n) 阶向量 $\bold{t}$，$\left[ \begin{array}{c} \bold{x} \\ \bold{z} \end{array} \right]$ 看作是 (m + n) 阶向量 $\bold{w}$，要求 $\bold{w} \ge \bold{0}$。

[S $\bold{t}$] = $\left[ \begin{array}{c} A & \bold{b} \\ C & \bold{d} \end{array} \right]$ 即是后续对偶单纯形算法中的单纯形表

---

<br>

当有如下矩阵形式后
$$
\left[ \begin{array}{c} A \\ C \end{array} \right] \bold{y} + 
\left[ \begin{array}{c} \bold{b} \\ \bold{d} \end{array} \right]
= \left[ \begin{array}{c} \bold{x} \\ \bold{z} \end{array} \right]
$$

那么原可行域 $\bold{F} = \{ \bold{x} | \bold{x} \ge \bold{0}, M\bold{x} + \bold{v} \ge \bold{0}\}$ 可以写成
$$
\left[ \begin{array}{c} I_n \\ M \end{array} \right] \bold{y} + 
\left[ \begin{array}{c} \bold{0} \\ \bold{v} \end{array} \right]
= \left[ \begin{array}{c} \bold{x} \\ \bold{z} \end{array} \right]
$$

初始单纯形表
$$
[S \ \bold{t}] = \left[ \begin{array}{cc} I_n & \bold{0} \\ M & \bold{v} \end{array} \right]
$$

---

<br>

初始单纯形表
$$
[S \ \bold{t}] = \left[ \begin{array}{cc} I_n & \bold{0} \\ M & \bold{v} \end{array} \right]
$$

S = [$\bold{s_1}$, ..., $\bold{s_n}$] 的列向量 $\bold{s_i}$ 都是 $[0,..., 0, s_{ij}, ...]^T$，$s_{ij} > 0$

我们称这种形式的向量为 lexico-positive，并且保证在后续对单纯形表做 pivot 操作后， S 的列向量始终保持这种性质。

<v-click>

假设我们始终保持这种性质，并且经过若干次 pivot 操作后， $\bold{t} \ge \bold{0}$，

$$
\left[ \begin{array}{c} A \\ C \end{array} \right] \bold{y} + 
\left[ \begin{array}{c} \bold{b} \\ \bold{d} \end{array} \right]
= \left[ \begin{array}{c} \bold{x} \\ \bold{z} \end{array} \right]
$$

令 $\bold{y} = \bold{0}$， $\bold{w} = \bold{t} \ge \bold{0}$，对应可行域 $\bold{F}$ 中一个元素 $\bold{x} = \bold{b}$。

</v-click>

---

现有 $\bold{t} \ge \bold{0}$，$\bold{x} = \bold{b}$ 为可行域 $\bold{F}$ 中一个元素，

$$
\left[ \begin{array}{c} A \\ C \end{array} \right] \bold{y} + 
\left[ \begin{array}{c} \bold{b} \\ \bold{d} \end{array} \right]
= \left[ \begin{array}{c} \bold{x} \\ \bold{z} \end{array} \right]
$$

证明其为可行域 $\bold{F}$ 中字典序最小的元素

$$
\begin{array}{c}
\bold{x} & = A\bold{y} + \bold{b} \\
& = \left[ \begin{array}{c} a_{11} \ a_{12} \ ... \ a_{1n} \\ a_{21} \ a_{22} \ ... \ a_{2n} \\ . \\ . \\ a_{n1} \ a_{n2} \ ... \ a_{nn} \end{array} \right] \left[ \begin{array}{c} y_1 \\ y_2 \\ . \\ . \\ y_n \end{array} \right] + \left[ \begin{array}{c} b_1 \\ b_2 \\ . \\ . \\ b_n \end{array} \right]
\end{array}
$$

因 $\red{\bold{y} \ge \bold{0}}$， 现有 $y_i = 0, 1 \le i \le n$， $\bold{y}$ 任意一维增加 $\Delta y_i > 0$， $\bold{x}$ 增加 $[a_{1i}\Delta y_i,a_{2i}\Delta y_i,...,a_{ni}\Delta y_i]^T$。

又有 $[a_{1i}, a_{2i}, ..., a_{ni}]^T, 1 \le i \le n$ 是 lexico-positive，因此 $[a_{1i}\Delta y_i,a_{2i}\Delta y_i,...,a_{ni}\Delta y_i]^T \gg \bold{0}$，

$[b_{1} + a_{1i}\Delta y_i, b_{2} + a_{2i}\Delta y_i, ..., b_{n} + a_{ni}\Delta y_i]^T \gg [b_{1}, b_{2}, ..., b_{n}]^T$，

因此 $\bold{b}$ 为 $\bold{F}$ 中字典序最小的元素。

<arrow v-click="1" x1="150" y1="355" x2="105" y2="380" color="red" width="1.5" arrowSize="1" />

<p v-after class="red absolute bottom-45 left-30 transform" style="color: red">还未说明</p>

---

$$
\left[ \begin{array}{c} A \\ C \end{array} \right] \bold{y} + 
\left[ \begin{array}{c} \bold{b} \\ \bold{d} \end{array} \right]
= \left[ \begin{array}{c} \bold{x} \\ \bold{z} \end{array} \right]
$$

对应的单纯形表

$$
\left[ \begin{array}{c|c} 
S_{11} \ \dots \ \dots \ S_{1n} & t_1 \\ 
\dots \ \dots \ \dots \ \dots \\ 
S_{n1} \ \dots \ \dots \ S_{nn} & t_n \\ 
S_{n+1 \ 1} \ \dots \ S_{n+1 \ n} & t_{n + 1} \\ 
\dots \ \dots \ \dots \ \dots \\ 
S_{m+n \ 1} \ \dots \ S_{m+n \ n} & t_{m + n} \\ 
\end{array} \right] = \left[ \begin{array}{c|c} 
a_{11} \ a_{12} \ ... \ a_{1n} & b_1 \\ 
\dots \ \dots \ \dots \ \dots \\ 
a_{n1} \ a_{n2} \ ... \ a_{nn} & b_n \\ 
c_{11} \ c_{12} \ ... \ c_{1n} & d_1 \\ 
\dots \ \dots \ \dots \ \dots \\ 
c_{m1} \ c_{m2} \ ... \ c_{mn} & d_m \\ 
\end{array} \right]
$$

选择 i 使得 $t_i < 0$，j 使得 $S_{ij} > 0$，有如下关系，其中 $w_i \ge 0$

$$
w_i = \sum_{k}S_{ik}y_k + t_i = \sum_{k \ne j}S_{ik}y_k + S_{ij}y_j + t_i
$$

将 $y_j$ 用 $y_k$($k \ne j$) 和 $w_i$ 表示

$$
y_j = \frac{w_i}{S_{ij}} - \sum_{k \ne j}\frac{S_{ik}}{S_{ij}}y_k - \frac{t_i}{S_{ij}}
$$

---

$$
\begin{array}{c}
w_m & = \sum_{k}S_{mk}y_k + t_m = \sum_{k \ne j}S_{mk}y_k + S_{mj}y_j + t_m \\
& = \sum_{k \ne j}S_{mk}y_k + S_{mj}(\frac{w_i}{S_{ij}} - \sum_{k \ne j}\frac{S_{ik}}{S_{ij}}y_k - \frac{t_i}{S_{ij}}) + t_m \\
& = \sum_{k \ne j}(S_{mk} - \frac{S_{ik}}{S_{ij}}S_{mj})y_k + \frac{S_{mj}}{S_{ij}}w_i + t_m - \frac{t_i}{S_{ij}}S_{mj}
\end{array}
$$

$$
\begin{array}{c}
w_i & = \sum_{k \ne j}(S_{ik} - \frac{S_{ik}}{S_{ij}}S_{ij})y_k + \frac{S_{ij}}{S_{ij}}w_i + t_i - \frac{t_i}{S_{ij}}S_{ij} = w_i
\end{array}
$$

$$
\left[ \begin{array}{c} w_1 \\ . \\ . \\ . \\ w_{m + n} \end{array} \right] = \left[ \begin{array}{c} 
S_{11} \ \dots \ \dots \ S_{1j} \ \dots \ \dots \ S_{1n} \\ 
\dots \ \dots \ \dots \ \dots \ \dots \ \dots \\ 
\dots \ \dots \ \dots \ \dots \ \dots \ \dots \\ 
S_{i1} \ \dots \ \dots \ S_{ij} \ \dots \ \dots \ S_{in} \\
\dots \ \dots \ \dots \ \dots \ \dots \ \dots \\ 
S_{m+n \ 1} \ \dots \ S_{m+n \ j} \ \dots \ S_{m+n \ n} \\ 
\end{array} \right]
\left[ \begin{array}{c} y_1 \\ . \\ y_j \\ . \\ . \\ y_n \end{array} \right] + \left[ \begin{array}{c} t_1 \\ . \\ . \\ t_i \\ . \\ t_{m + n} \end{array} \right] \xrightarrow[\text{$y_j$ leaves}]{\text{$w_i$ enters}} \\
\left[ \begin{array}{c} 
S_{11} - \frac{S_{i1}}{S_{ij}}S_{1j} \ \ \dots \ \dots \ \dots \ \frac{S_{1j}}{S_{ij}} \ \dots \ \dots \ \dots \ S_{1n} - \frac{S_{in}}{S_{ij}}S_{1j} \\ 
\dots \ \dots \ \dots \ \dots \ \dots \ \dots \ \dots \ \dots \ \dots \ \dots \ \dots \ \dots \\ 
\dots \ \dots \ \dots \ \dots \ \dots \ \dots \ \dots \ \dots \ \dots \ \dots \ \dots \ \dots \\ 
0 \ \dots \ \dots \ \dots \ \dots \ \dots \ 1 \ \dots \ \dots \ \dots \ \dots \ \dots \ 0 \\
\dots \ \dots \ \dots \ \dots \ \dots \ \dots \ \dots \ \dots \ \dots \ \dots \ \dots \ \dots  \\ 
S_{m+n \ 1} - \frac{S_{i \ 1}}{S_{ij}}S_{m + n \ j} \ \dots \ \dots \ \frac{S_{m + n \ j}}{S_{ij}} \ \dots \ \dots \ S_{m + n \ n} - \frac{S_{in}}{S_{ij}}S_{m + n \ j} \\ 
\end{array} \right]
\left[ \begin{array}{c} y_1 \\ . \\ \red{w_i} \\ . \\ . \\ y_n \end{array} \right] + \left[ \begin{array}{c} t_1 - \frac{t_i}{S_{ij}}S_{1j} \\ . \\ . \\ 0 \\ . \\ t_{m + n} - \frac{t_i}{S_{ij}}S_{m + n \ j} \end{array} \right]
$$

<arrow v-click="1" x1="730" y1="420" x2="705" y2="435" color="red" width="1.5" arrowSize="1" />

<p v-after class="red absolute bottom-28 right-50 transform" style="color: red">y* > 0</p>

<v-click>

<p v-after class="red absolute top-20 left-20 transform" style="color: red">(i,j)-pivot</p>

</v-click>

---

### 证明 S 的列向量在 pivot 操作后仍保持 lexico-positive 性质

因为 $S_{ij}$ 为正数，所以 $[\frac{S_{1j}}{S_{ij}}, \dots, 1, \dots, \frac{S_{m + n \ j}}{S_{ij}}]^T$ 仍然是 lexico-positive 的

要使 $[S_{1k} - \frac{S_{ik}}{S_{ij}}S_{1j}, \dots, 0, \dots, S_{m + n \ k} - \frac{S_{ik}}{S_{ij}}S_{m + n \ j}]^T$ 是 lexico-positive，需要 $[\frac{S_{1j}}{S_{ij}},\dots,\frac{S_{m+n \ j}}{S_{ij}}]^T$ 是 lexico-minimal 的。

假设做 (2, j)-pivot，需要确定 j：
$\left[ \begin{array}{c} 3 \\ 2 \\ 2 \end{array} \right] \ll \left[ \begin{array}{c} 4 \\ 4 \\ 4 \end{array} \right], \left[ \begin{array}{c} 3/2 \\ 2/2 \\ 2/2 \end{array} \right] \gg \left[ \begin{array}{c} 4/4 \\ 4/4 \\ 4/4 \end{array} \right]$

证明：

1) $S_{ik} < 0$，$S_{mk}' = S_{mk} - \frac{S_{ik}}{S_{ij}}S_{mj} = S_{mk} + \frac{|S_{ik}|}{|S_{ij}|}S_{mj}$，即一个 lexico-positive 的列向量加上另一个 lexico-positive 的列向量，因此新的列向量同样也是 lexico-positive。

2) $S_{ik} = 0$，列向量不变。

3) $S_{ik} > 0$，因为 $[\frac{S_{1k}}{S_{ik}},\dots,\frac{S_{m+n \ k}}{S_{ik}}]^T \gg [\frac{S_{1j}}{S_{ij}},\dots,\frac{S_{m+n \ j}}{S_{ij}}]^T$，考察第 k 列向量中的 $S_{ak}$，是第一个满足 $\frac{S_{ak}}{S_{ik}} > \frac{S_{aj}}{S_{ij}}$ 的元素， $S_{ak}' = S_{ak} - \frac{S_{ik}}{S_{ij}}S_{aj} = S_{ak} - \frac{S_{ik}}{S_{ij}}S_{aj} = \frac{S_{ak}S_{ij} - S_{ik}S_{aj}}{S_{ij}} > 0$，列向量仍然是 lexico-positive。

---

<br>

再来看 $\bold{t}$，$\bold{t}' = \bold{t} - \frac{t_i}{S_{ij}}s_{j}$，因为 $t_i < 0, S_{ij} > 0$，$s_{j}$ 是 lexico-positive 的，因此 $\bold{t}$ 是字典序单调递增。 $S_{1j} \ge 0$，则 $t_1' = t_1 - \frac{t_i}{S_{ij}}S_{1j}$，$t_1$ 单调递增。

回顾问题：

$$
\bold{F} = \{ A\bold{y} + \bold{b} | \bold{x} = A\bold{y} + \bold{b} \ge \bold{0}, \bold{z} = C\bold{y} + \bold{d} \ge \bold{0}, \bold{y} \ge \bold{0} \}
$$

若问题的可行域非空，则有字典序最小的解 $\bold{u} = [u_1, \dots, u_n]^T$，

$$
\sum_{j} S_{1j}y_{j} + t_1 \ge u_1 \ge t_1
$$

---

### 算法收敛性

<v-click>

面对原可行域
$$
\left[ \begin{array}{c} I_n \\ M \end{array} \right] \bold{y} + 
\left[ \begin{array}{c} \bold{0} \\ \bold{v} \end{array} \right]
= \left[ \begin{array}{c} \bold{x} \\ \bold{z} \end{array} \right]
$$
到底做了什么？

</v-click>

<v-click>

对
$$
w_i = \sum_{k}S_{ik}y_k + t_i = \sum_{k \ne j}S_{ik}y_k + S_{ij}y_j + t_i
$$

进行变换，将 $y_j$ 用 $y_k$($k \ne j$) 和 $w_i$ 表示

$$
y_j = \frac{w_i}{S_{ij}} - \sum_{k \ne j}\frac{S_{ik}}{S_{ij}}y_k - \frac{t_i}{S_{ij}}
$$

</v-click>

<v-click>

也即将原本用 $x_1, \dots, x_n$ 表示的 $\bold{w}$，一部分用 $x_1, \dots, x_n$，一部分用 $z_1, \dots, z_m$ 表示，也就是换基。只要基（basic）确定，$[S, t]$ 也确定了，基与 $\bold{t}$ 一一对应。又因为 $\bold{t}$ 单调递增，最多有 $C_{m + n}^{n}$ 个基，因此算法收敛。

</v-click>

---

求解 $R^n$ 上的可行域中字典序最小元素的算法如下：

- 根据问题约束构建初始的单纯形表
$$
[S \ \bold{t}] = \left[ \begin{array}{cc} I_n & \bold{0} \\ M & \bold{v} \end{array} \right]
$$

- 取 i 使得 $t_i < 0$，如果 $\forall i, t_i \ge 0$ 进入步骤 3

  - 取 j 使得 $S_{ij} > 0$ 且 $[\frac{S_{1j}}{S_{ij}},\dots,\frac{S_{m+n \ j}}{S_{ij}}]^T$ 字典序最小，进行 (i,j)-pivot 操作。如果 $\forall j, S_{ij} \le 0$，则有 $\sum_{j}S_{ij}y_{j} + t_i = w_i < 0$，不满足约束，原可行域为空。

- 如果当前单纯形表中仍存在 $t_i < 0$，重复步骤 2。否则令 $\bold{y} = \bold{0}$，字典序最小的元素 $\bold{x} = \bold{b}$。

---

<br>

<style>
  ul {
    font-size: 30px;
  }

  li {
    font-size: 26px;
    margin: 15px 0;
  }

  li.transparent {
    color: #9ea7b3de
  }
</style>

<ul>
  整数线性规划问题（Integer linear programming）
  <li class="transparent">单纯形算法（Simplex method）</li>


  <li class="transparent">整数单纯形算法（Simplex method + Gomory cut）</li>
  
  <li class="transparent">字典序最小问题（Lexicographical minimum）</li>

  <li>整数字典序最小问题（Lexicographical minimum + Gomory cut）</li>
</ul>

---

# 问题定义

<br>

$Z^n$ 上的字典序最小解：给定 m $\times$ n 的矩阵 M，m 维的向量 $\bold{v}$

令 $\bold{F} = \{ \bold{x} | \bold{x} \in N^n, M\bold{x} + \bold{v} \ge \bold{0} \}$

集合 $\bold{F}$ 为问题的可行域，判定集合 $\bold{F}$ 是否为空，若不为空则求出集合中字典序最小的元素。

不失一般性地，假设 M 和 $\bold{v}$ 中的元素都是整数，则 $M \bold{x} + \bold{v}$ 是整数向量。

因此原可行域抽象成

$$
\bold{F} = \{ A\bold{y} + \bold{b} | \bold{x} = A\bold{y} + \bold{b} \in N^n, \bold{z} = C\bold{y} + \bold{d} \in N^m, \red{\bold{y} \in N^n} \}
$$

<br>

---

针对可行域
$$
\bold{F} = \{ A\bold{y} + \bold{b} | \bold{x} = A\bold{y} + \bold{b} \in N^n, \bold{z} = C\bold{y} + \bold{d} \in N^m, \red{\bold{y} \in N^n} \}
$$

如果还是完全照搬上一节的算法，（算法收敛后）得到 $\bold{t} = \left[ \begin{array}{c} \bold{b} \\ \bold{d} \end{array} \right]$ 是可行域

$$
\overline{\bold{F}} = \{ A\bold{y} + \bold{b} | \bold{x} = A\bold{y} + \bold{b} \ge \bold{0}, \bold{z} = C\bold{y} + \bold{d} \ge \bold{0}, \red{\bold{y} \ge \bold{0}} \}
$$

字典序最小值，且无法保证 $\bold{b}$ 是整数。对于 $\bold{F}$ 的字典序最小值 $\bold{u}$，我们知道 $\bold{u} \in \overline{\bold{F}}$，因此有 $\bold{b} \ll \bold{u}$

<v-click>

怎么处理呢？

</v-click>

<v-click>

通过 Gomory cut 引入一个新的约束，在排除掉 $\bold{b}$ 这样的全局非整数的字典序最小解的同时，保留所有可行的整数解。

</v-click>

---

## Gomory cut

选取第一个不是整数的 $b_i$ 对应的 A 中的行，如果不存在这样的行，则 $\bold{b}$ 是整数向量，且为原可行域的整数字典序最小解。令 D 为 $S_{ij}$ 和 $b_i$ 的最小公分母，若有

$$
\sum_{j} S_{ij}y_j + t_i \in N
$$

则

$$
\sum_{j} (DS_{ij})y_j + (Dt_i) \equiv 0 \mod D
$$

对上式进行取模运算

$$
\sum_{j} ((DS_{ij}) \% D)y_j \equiv (-Dt_i) \% D \mod D
$$

进一步有

$$
\sum_{j} ((DS_{ij}) \% D)y_j = (-Dt_i) \% D + kD(k \ge 0)
$$

---

$$
\sum_{j} ((DS_{ij}) \% D)y_j = (-Dt_i) \% D + kD(k \ge 0)
$$

根据 $k \ge 0$，可以将上式写成

$$
\sum_{j} ((DS_{ij}) \% D)y_j - (-Dt_i) \% D \ge 0
$$

并且知道

$$
\sum_{j} \frac{((DS_{ij}) \% D)}{D}y_j - \frac{(-Dt_i) \% D}{D} = k
$$

是一个非负整数，因此可以加入

$$
\sum_{j} \frac{((DS_{ij}) \% D)}{D}y_j - \frac{(-Dt_i) \% D}{D} \in N
$$

作为 cut

---

针对新的约束

$$
\sum_{j} \frac{((DS_{ij}) \% D)}{D}y_j - \frac{(-Dt_i) \% D}{D} \in N
$$

其常数项为负，可以进一步 pivot。

<v-click>

我们根据

$$
\sum_{j} S_{ij}y_j + t_i \in N
$$

推导出约束将非整数的字典序最小解排除在外，但原可行域的整数可行解都保留。继续 pivot 可能发现原可行域为空，即原问题无整数字典序最小解，或者不断加入 Gomory cut 最后得到一个整数解，通过类似上一节的证明证得是整数字典序最小解。

</v-click>

---

### 算法收敛性

有原可行域
$$
\bold{F} = \{ \bold{x} | \bold{x} \in N^n, M\bold{x} + \bold{v} \in N^n \}
$$

令 $\bold{F}_n$ 为原问题加入 n 个 cut 后的可行域

$$
\bold{F}_n = \{ A^{(n)}\bold{y} + \bold{b}^{(n)} | \bold{x} = A^{(n)}\bold{y} + \bold{b}^{(n)} \in N^n, \bold{z} = C^{(n)}\bold{y} + \bold{d}^{(n)} \in N^m, \red{\bold{y} \in N^n} \}
$$

令 $\bold{F}_n^*$ 为加入第 n 个 cut 并且 pivot 之后的可行域表示形式。

---

对于新加入的一个 cut

$$
\sum_{j} \frac{((DS_{ij}) \% D)}{D}y_j - \frac{(-Dt_i) \% D}{D} \in N
$$

令 $\sigma(n)$ 为新加入的第 n 个 cut 对应的约束的行号，则第 n 个 cut 的常数项为 $-\frac{(-D^{(n)}b_{\sigma(n)}^{(n)}) \% D^{(n)}}{D^{(n)}}$，进行 pivot 时，其进行 pivot 操作时对应的 $S_{ij}$ 为 $\frac{(D^{(n)}A_{\sigma(n)j}^{(n)}) \% D^{(n)}}{D^{(n)}}$

则有

$$
b_{\sigma(n)}^{'(n)} = b_{\sigma(n)}^{(n)} + \frac{(-D^{(n)}b_{\sigma(n)}^{(n)}) \% D^{(n)}}{D^{(n)}} \frac{D^{(n)}}{(D^{(n)}A_{\sigma(n)j}^{(n)}) \% D^{(n)}} A_{\sigma(n)j}^{(n)}
$$

---

$$
b_{\sigma(n)}^{'(n)} = b_{\sigma(n)}^{(n)} + \frac{(-D^{(n)}b_{\sigma(n)}^{(n)}) \% D^{(n)}}{D^{(n)}} \frac{D^{(n)}}{(D^{(n)}A_{\sigma(n)j}^{(n)}) \% D^{(n)}} A_{\sigma(n)j}^{(n)}
$$

已知 $b_{\sigma(n)}^{(n)}$ 和 $A_{\sigma(n)j}^{(n)}$ 为正数，且有

$$
(D^{(n)}A_{\sigma(n)j}^{(n)}) \% D^{(n)} \le D^{(n)}A_{\sigma(n)j}^{(n)}
$$

可得

$$
b_{\sigma(n)}^{'(n)} \ge b_{\sigma(n)}^{(n)} + \frac{(-D^{(n)}b_{\sigma(n)}^{(n)}) \% D^{(n)}}{D^{(n)}} = \frac{b_{\sigma(n)}^{(n)}D^{(n)} + (-D^{(n)}b_{\sigma(n)}^{(n)}) \% D^{(n)}}{D^{(n)}}
$$

令 $b_{\sigma(n)}^{(n)} = q(n) - \frac{r(n)}{D^{(n)}}$，$q(n)$ 和 $r(n)$ 都是整数，且 $0 \le r(n) < D^{(n)}$，将 $b_{\sigma(n)}^{(n)}$ 带入上式有

$$
b_{\sigma(n)}^{'(n)} \ge q(n) > b_{\sigma(n)}^{(n)}
$$

因为 $\sigma(n)$ 的选取满足 $b_{\sigma(n)}^{(n)}$ 非整数，由上式可知，每次由 $\sigma(n)$-行获得新的 cut 并进行 pivot 操作后，$b_{\sigma(n)}^{'(n)}$ 和 $b_{\sigma(n)}^{(n)}$ 间必定会有一个整数 $\{b_{\sigma(n)}^{(n)}\}$

---

现在考虑 $b_1$ 所在的第一行，假设当前引入的 cut 不对应第一行，则 $b_1$ 为整数。
- 令 j 为下一个 pivot 操作对应的列，如果 $S_{1j} = 0$，则 $b_1$ pivot 过程中不变。
- 否则 $S_{1j} > 0$，$b_1$ 增加。
  - 如果增加的部分是整数，那么 $b_1$ 至少增加了 1。
  - 否则下一个 cut 的就会用到第一行，$b_1$ 还是会超过 $b_1 + 1$

我们定义一类 cut 为 1-cut，满足：

- 选取第一行来生成新的 cut
- 在做插入当前 cut 后的下一个 pivot 时，$S_{1j} > 0$，j 是做 pivot 操作时对应的列号

假设第 n 个引入的 cut 是 1-cut，则 $[b_1^{(n)}, b_1^{(n+1)}]$ 区间内至少有一个整数，$[b_1^{(i)}, b_1^{(i+1)}]$ 区间内的这些整数构成了一个严格单调递增的序列，假设前 n 个 cut 中有 $K_n$ 个 1-cut，分别为 $i_1, \dots, i_{K_n}$，则有 $1 \le i_1 < i_2 < \dots < i_{K_n} \le n$，则有

$$
b_1^{(n)} - b_1^{(0)} \ge K_n - 1
$$

---

原可行域
$$
\bold{F} = \{ \bold{x} | \bold{x} \in N^n, M\bold{x} + \bold{v} \in N^n \}
$$

$\bold{F}_n$ 为原问题加入 n 个 cut 后的可行域

$$
\bold{F}_n = \{ A^{(n)}\bold{y} + \bold{b}^{(n)} | \bold{x} = A^{(n)}\bold{y} + \bold{b}^{(n)} \in N^n, \bold{z} = C^{(n)}\bold{y} + \bold{d}^{(n)} \in N^m, \red{\bold{y} \in N^n} \}
$$

已知 $\bold{F}$ 的字典序最小解为 $\bold{u}$ 并且 $\bold{u} \in F_n$，因此存在 $\bold{y}$

$$
\bold{u} = A^{(n)}\bold{y} + \bold{b}^{(n)}, \bold{y} \ge \bold{0}
$$

因为 $A^{(n)}$ 是 lexico-positive，有 $\bold{b}^{(n)} \ll \bold{u}$，并且 $\bold{b}_1^{(n)} \le \bold{u}_1$，可知 1-cut 的数目是有限的，因此

$$
\exists N_1 \text{, s.t.} \forall n \ge N_1, b_1^{(n)} = b_1^{(N_1)}
$$

并且 $b_1^{(N_1)}$ 是整数。因此对于 $N_1$ 之后的 cut，pivot 只会在 $S_{1j} = 0$ 的列上进行。

---

对于第 i 行， $J_i^+$（$J_i^0$， $J_i^-$）对应列号 j 的集合满足 $S_{ij} > 0$（$S_{ij} = 0$，$S_{ij} < 0$），因此有

$$
\forall j \in J_1^+: 0 \le y_j \le  \frac{\bold{u}_1 - \bold{b}_1^{(N_1)}}{A_{1j}^{(N_1)}}
$$

并且对于 $N_1$ 之后的 cut，上式始终成立，因为后续的 pivot 只在 $S_{1j} = 0$ 的列上进行，因此第一行 $S_{1j}$ 后续不会变化。

考察第二行，为了保证 S 的所有列向量均是 lexico-positive，可有 $\forall j \in J_1^0: A_{2j}^{(N_1)} \ge 0$，因此 $J_2^- \subseteq J_1^+$

$$
\bold{b}_2^{(N_1)} = \bold{u}_2 - \sum_{j} A_{2j}^{(N_1)}y_j \le \bold{u}_2 + \sum_{j \in J_2^-}(-A_{2j}^{(N_1)})y_j \le \bold{u}_2 + \sum_{j \in J_2^-}(-A_{2j}^{(N_1)})\frac{\bold{u}_1 - \bold{b}_1^{(N_1)}}{A_{1j}^{(N_1)}}
$$

因此 2-cut 这一类的 cut 数量也是有限的，参照上述流程可得

$$
\exists N_2 \text{, s.t.} \forall n \ge N_2, b_2^{(n)} = b_2^{(N_2)}
$$

且 $b_2^{(N_2)}$ 为整数，依次得到 $N_1, N_2, \dots, N_n$ 用以刻画 n-cut 的数目，最终得到 $\bold{b}$ 为整数向量，算法结束。

---

求解 $Z^n$ 上的可行域中字典序最小元素的算法如下：

- 根据问题约束构建初始的单纯形表
$$
[S \ \bold{t}] = \left[ \begin{array}{cc} I_n & \bold{0} \\ M & \bold{v} \end{array} \right]
$$

- 按照 $R^n$ 上求解可行域中字典序最小元素的算法求出（非整数的）字典序最小元素

- 取 i 使得 i 为不满足 $b_i \in N$ 的最小值。如果 $\forall i, b_i \in N$ 则得到整数字典序最小解 $\bold{b}$，否则根据第 i 行对应的约束加入相应的 Gomory cut，进入步骤 2

---
layout: image-right
image: https://source.unsplash.com/collection/94734566/1920x1080
---

# Code

Use code snippets and get the highlighting directly![^1]

```ts {all|2|1-6|9|all}
interface User {
  id: number
  firstName: string
  lastName: string
  role: string
}

function updateUser(id: number, update: User) {
  const user = getUser(id)
  const newUser = {...user, ...update}  
  saveUser(id, newUser)
}
```

<arrow v-click="3" x1="400" y1="420" x2="230" y2="330" color="#564" width="3" arrowSize="1" />

[^1]: [Learn More](https://sli.dev/guide/syntax.html#line-highlighting)

<style>
.footnotes-sep {
  @apply mt-20 opacity-10;
}
.footnotes {
  @apply text-sm opacity-75;
}
.footnote-backref {
  display: none;
}
</style>

---

# Components

<div grid="~ cols-2 gap-4">
<div>

You can use Vue components directly inside your slides.

We have provided a few built-in components like `<Tweet/>` and `<Youtube/>` that you can use directly. And adding your custom components is also super easy.

```html
<Counter :count="10" />
```

<!-- ./components/Counter.vue -->
<Counter :count="10" m="t-4" />

Check out [the guides](https://sli.dev/builtin/components.html) for more.

</div>
<div>

```html
<Tweet id="1390115482657726468" />
```

<Tweet id="1390115482657726468" scale="0.65" />

</div>
</div>


---
class: px-20
---

# Themes

Slidev comes with powerful theming support. Themes can provide styles, layouts, components, or even configurations for tools. Switching between themes by just **one edit** in your frontmatter:

<div grid="~ cols-2 gap-2" m="-t-2">

```yaml
---
theme: default
---
```

```yaml
---
theme: seriph
---
```

<img border="rounded" src="https://github.com/slidevjs/themes/blob/main/screenshots/theme-default/01.png?raw=true">

<img border="rounded" src="https://github.com/slidevjs/themes/blob/main/screenshots/theme-seriph/01.png?raw=true">

</div>

Read more about [How to use a theme](https://sli.dev/themes/use.html) and
check out the [Awesome Themes Gallery](https://sli.dev/themes/gallery.html).

---
preload: false
---

# Animations

Animations are powered by [@vueuse/motion](https://motion.vueuse.org/).

```html
<div
  v-motion
  :initial="{ x: -80 }"
  :enter="{ x: 0 }">
  Slidev
</div>
```

<div class="w-60 relative mt-6">
  <div class="relative w-40 h-40">
    <img
      v-motion
      :initial="{ x: 800, y: -100, scale: 1.5, rotate: -50 }"
      :enter="final"
      class="absolute top-0 left-0 right-0 bottom-0"
      src="https://sli.dev/logo-square.png"
    />
    <img
      v-motion
      :initial="{ y: 500, x: -100, scale: 2 }"
      :enter="final"
      class="absolute top-0 left-0 right-0 bottom-0"
      src="https://sli.dev/logo-circle.png"
    />
    <img
      v-motion
      :initial="{ x: 600, y: 400, scale: 2, rotate: 100 }"
      :enter="final"
      class="absolute top-0 left-0 right-0 bottom-0"
      src="https://sli.dev/logo-triangle.png"
    />
  </div>

  <div 
    class="text-5xl absolute top-14 left-40 text-[#2B90B6] -z-1"
    v-motion
    :initial="{ x: -80, opacity: 0}"
    :enter="{ x: 0, opacity: 1, transition: { delay: 2000, duration: 1000 } }">
    Slidev
  </div>
</div>

<!-- vue script setup scripts can be directly used in markdown, and will only affects current page -->
<script setup lang="ts">
const final = {
  x: 0,
  y: 0,
  rotate: 0,
  scale: 1,
  transition: {
    type: 'spring',
    damping: 10,
    stiffness: 20,
    mass: 2
  }
}
</script>

<div
  v-motion
  :initial="{ x:35, y: 40, opacity: 0}"
  :enter="{ y: 0, opacity: 1, transition: { delay: 3500 } }">

[Learn More](https://sli.dev/guide/animations.html#motion)

</div>

---

# LaTeX

LaTeX is supported out-of-box powered by [KaTeX](https://katex.org/).

<br>

Inline $\sqrt{3x-1}+(1+x)^2$

Block
$$
\begin{array}{c}

\nabla \times \vec{\mathbf{B}} -\, \frac1c\, \frac{\partial\vec{\mathbf{E}}}{\partial t} &
= \frac{4\pi}{c}\vec{\mathbf{j}}    \nabla \cdot \vec{\mathbf{E}} & = 4 \pi \rho \\

\nabla \times \vec{\mathbf{E}}\, +\, \frac1c\, \frac{\partial\vec{\mathbf{B}}}{\partial t} & = \vec{\mathbf{0}} \\

\nabla \cdot \vec{\mathbf{B}} & = 0

\end{array}
$$

<br>

[Learn more](https://sli.dev/guide/syntax#latex)

---

# Diagrams

You can create diagrams / graphs from textual descriptions, directly in your Markdown.

<div class="grid grid-cols-2 gap-10 pt-4 -mb-6">

```mermaid {scale: 0.9}
sequenceDiagram
    Alice->John: Hello John, how are you?
    Note over Alice,John: A typical interaction
```

```mermaid {theme: 'neutral', scale: 0.8}
graph TD
B[Text] --> C{Decision}
C -->|One| D[Result 1]
C -->|Two| E[Result 2]
```

</div>

[Learn More](https://sli.dev/guide/syntax.html#diagrams)


---
layout: center
class: text-center
---

# Learn More

[Documentations](https://sli.dev) · [GitHub](https://github.com/slidevjs/slidev) · [Showcases](https://sli.dev/showcases.html)
