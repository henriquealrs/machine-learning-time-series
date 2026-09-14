# Dimensional analysis of the vehicle data

## Purpose

`Mackenzie_Aprendizagem_Maquina.py` reads vehicle measurements and physical
metadata from `data/1000GD.xlsx`. It uses the Buckingham $\Pi$ theorem to build
dimensionless combinations of those measurements. For each combination, it
plots a scatter graph against fuel rate and adds a linear trend line.

The script does not train a general machine-learning model. Its statistical
step is a separate ordinary least-squares linear regression for each generated
$\Pi$ group.

## Workbook structure

The first worksheet contains the observations. Each row is one observation and
each column is a measured quantity, such as vehicle speed, engine speed,
torque, fuel rate, distance, or axle weight.

The second worksheet describes the physical quantities. Its important columns
are:

| Column | Meaning |
| --- | --- |
| `Alias` | Short mathematical name, such as `v`, `tau`, or `m` |
| `Parameter` | Corresponding column name in the measurement worksheet |
| `SI` | Display form of the SI unit |
| `M`, `L`, `T`, `K` | Exponents of mass, length, time, and temperature |
| `Repeatable` | An `ok` value identifies a repeating variable |

A quantity $q$ with dimension exponents $a$, $b$, $c$, and $d$ is represented
as

$$
[q] = M^a L^b T^c K^d.
$$

For example, torque has SI dimensions

$$
[\tau] = M L^2 T^{-2},
$$

so its metadata row contains the exponent vector

$$
\begin{bmatrix}1 & 2 & -2 & 0\end{bmatrix}.
$$

The code identifies fundamental dimensions as the one-character column names
in the metadata sheet. With the current workbook they are $M$, $L$, $T$, and
$K$.

## Repeating variables

The workbook marks these three quantities as repeating variables:

| Alias | Quantity | Dimensions |
| --- | --- | --- |
| $\tau$ | Engine torque | $M L^2 T^{-2}$ |
| $x$ | Distance | $L$ |
| $m$ | Axle mass | $M$ |

The repeating variables must be dimensionally independent and must span all
dimensions needed to nondimensionalize the target quantities. Temperature
$K$ is absent from all three repeating variables, so the script removes the
all-zero temperature row from the matrix.

After that removal, the dimensional matrix is

$$
A =
\begin{bmatrix}
1 & 0 & 1 \\
2 & 1 & 0 \\
-2 & 0 & 0
\end{bmatrix},
$$

where the rows represent $M$, $L$, and $T$, while the columns represent
$\tau$, $x$, and $m$.

## Constructing a dimensionless group

For a target variable $q$, the script looks for a group of the form

$$
\Pi = \tau^{p_\tau} x^{p_x} m^{p_m} q.
$$

Let the target's dimensional-exponent vector be

$$
b =
\begin{bmatrix}
b_M \\
b_L \\
b_T
\end{bmatrix}
$$

and let the unknown repeating-variable powers be

$$
p =
\begin{bmatrix}
p_\tau \\
p_x \\
p_m
\end{bmatrix}.
$$

For $\Pi$ to be dimensionless, every resulting fundamental-dimension exponent
must equal zero:

$$
A p + b = 0.
$$

Therefore the program solves

$$
A p = -b.
$$

`numpy.linalg.solve` performs this calculation. It requires $A$ to be square
and nonsingular. The current three repeating variables and three active
dimensions satisfy those requirements.

Before solving, the script also checks dimensions removed from $A$. If a target
uses a removed dimension, then the repeating variables cannot cancel it and no
$\Pi$ group is produced for that target.

## Groups produced by the current metadata

The metadata produces the following groups, up to floating-point
representation:

### Vehicle speed

Because $[v] = L T^{-1}$, the solution is

$$
p =
\begin{bmatrix}
-\frac{1}{2} \\
0 \\
\frac{1}{2}
\end{bmatrix},
$$

and therefore

$$
\Pi_1 = v\,\tau^{-1/2}m^{1/2}.
$$

### Engine speed

Because $[n] = T^{-1}$, the result is

$$
\Pi_2 = n\,\tau^{-1/2}x m^{1/2}.
$$

### Fuel rate

Using $[fr] = L^3T^{-1}$ gives

$$
\Pi_3 = fr\,\tau^{-1/2}x^{-2}m^{1/2}.
$$

### Altitude

Since altitude and distance both have dimension $L$, their ratio is
dimensionless:

$$
\Pi_4 = z x^{-1} = \frac{z}{x}.
$$

The script retains zero powers in its internal group descriptions. A factor
such as $\tau^0$ or $m^0$ is equal to one and does not change the mathematical
result.

## Mapping the mathematics to dataframe columns

The matrix calculations use concise aliases. Before evaluating the groups, the
script replaces each alias with the full `Parameter` name from the metadata.
For example,

$$
v\,\tau^{-1/2}m^{1/2}
$$

becomes a dataframe calculation using `TachographVehicleSpeed`,
`ActualEnginePercentTorque[percent]`, and `D_AxleWeight[kg]`.

For every $\Pi_i$, a new dataframe column is built by multiplying the powered
source columns:

$$
\text{data}[\Pi_i] = \prod_j \left(\text{data}[q_j]\right)^{p_j}.
$$

This calculation is performed in place, so the measurement dataframe gains
columns named `Pi1`, `Pi2`, and so on as execution proceeds.

## Regression and plotting

The response variable is selected by the constant `Y_AXIS_INDEX`. Its current
value is $6$, corresponding to `EngineFuelRate` in the measurement worksheet.
For each successfully evaluated group, the script uses rows from
`INITIAL_ROW` through `FINAL_ROW`, inclusive under pandas label slicing.

It fits the straight line

$$
y = \beta_0 + \beta_1 \Pi_i,
$$

where $y$ is fuel rate, $\beta_0$ is the intercept, and $\beta_1$ is the slope.
The reported statistic is

$$
R^2 = r^2,
$$

where $r$ is the correlation coefficient returned by
`scipy.stats.linregress`. Each figure contains the original observations and
the fitted line.

## Numerical and input assumptions

- Every metadata `Parameter` used by a generated group must exactly match a
  measurement-sheet column name. The current metadata includes
  `GPS_Altitude[m]`, but the current measurement sheet does not. Consequently,
  evaluation of $\Pi_4$ retains the original behavior of raising a `KeyError`.
- Fractional negative powers such as $\tau^{-1/2}$ require suitable nonzero,
  nonnegative input values. Zero or negative torque can create infinite or
  missing values.
- `scipy.stats.linregress` is called directly without cleaning non-finite
  values. Such values can therefore cause the printed $R^2$ to be `nan`.
- The repeating-variable matrix must be square and nonsingular. Otherwise,
  `numpy.linalg.solve` raises a linear-algebra error.
- The metadata dimension columns and their exponents must consistently describe
  the SI units. The program trusts these declarations rather than deriving
  dimensions from the text in the `SI` column.

## Code organization

The refactored script separates the workflow into focused functions:

1. `load_workbook` reads the two worksheets.
2. `find_fundamental_dimensions` discovers the dimension columns.
3. `parse_parameter_specs` creates typed parameter records.
4. `select_repeatable_parameters` reads the repeating-variable choices.
5. `build_dimensional_matrix` constructs and reduces $A$.
6. `solve_pi_groups` solves $A p = -b$ for each target.
7. `replace_aliases_with_column_names` connects formulas to measurement data.
8. `calculate_pi_column` evaluates one group.
9. `plot_pi_regression` performs and plots one regression.
10. `main` coordinates the complete sequence.

The `if __name__ == "__main__"` guard means the functions and typed data
classes can also be imported for tests or reuse without immediately running
the workbook analysis.
