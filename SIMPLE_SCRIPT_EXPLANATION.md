# Simple explanation of the dimensional-analysis script

## What the script does

In simple terms, `Mackenzie_Aprendizagem_Maquina.py` tries to find normalized
combinations of vehicle measurements and see how each combination relates to
fuel consumption.

A normalized value is useful because it makes different physical measurements
easier to compare. In this script, the normalized values are dimensionless:
their units cancel completely. This means that a result does not depend on
whether a distance was originally expressed in meters or another compatible
unit, provided all input units are converted consistently.

The script follows these steps:

1. It reads vehicle measurements from the first Excel worksheet.
2. It reads variable names, units, and physical dimensions from the second
   worksheet.
3. It uses torque, distance, and mass as common building blocks called
   repeating variables.
4. It combines those building blocks with vehicle speed, engine speed, fuel
   rate, and altitude so that all physical units cancel.
5. It creates new dataframe columns called `Pi1`, `Pi2`, `Pi3`, and `Pi4`.
6. It plots each calculated $\Pi$ value against `EngineFuelRate`.
7. It draws a red straight line representing the overall linear trend.

## How to read the plots

Every intended plot has the same basic structure:

- Each dot represents one row from the Excel measurement worksheet.
- The horizontal axis contains one calculated $\Pi$ value.
- The vertical axis contains the original `EngineFuelRate` measurement.
- The red line is the straight line that best fits all the dots.
- The printed $R^2$ value describes how closely the dots follow that line.

An $R^2$ value near $1$ indicates a strong linear relationship. A value near
$0$ indicates a weak linear relationship. A result of `nan` means the
calculation contains values that the regression cannot use, such as missing or
infinite values.

These plots show associations in the data. They do not prove that the value on
the horizontal axis causes fuel consumption to change.

## Plot 1: normalized vehicle speed

The first horizontal-axis value is

$$
\Pi_1 = v\sqrt{\frac{m}{\tau}},
$$

where:

- $v$ is vehicle speed;
- $m$ is axle mass; and
- $\tau$ is engine torque.

This value is approximately the vehicle speed compared with a characteristic
speed determined by mass and torque.

In simple terms, the plot asks:

> Does fuel rate change predictably as normalized vehicle speed increases?

## Plot 2: normalized engine speed

The second horizontal-axis value is

$$
\Pi_2 = nx\sqrt{\frac{m}{\tau}},
$$

where:

- $n$ is engine speed;
- $x$ is total vehicle distance;
- $m$ is axle mass; and
- $\tau$ is engine torque.

This combines engine speed with distance, mass, and torque.

In simple terms, the plot asks:

> Does fuel rate have a linear relationship with this normalized
> engine-operating value?

Because $x$ is total accumulated distance, its magnitude may strongly
influence the result. Whether this particular combination is physically useful
depends on what the distance measurement represents in the experiment.

## Plot 3: normalized fuel rate

The third horizontal-axis value is

$$
\Pi_3 = fr\,x^{-2}\sqrt{\frac{m}{\tau}},
$$

where $fr$ is fuel rate.

In simple terms, the plot asks:

> How does the original fuel rate relate to a normalized version of itself?

This plot requires careful interpretation because fuel rate appears on both
axes:

- The horizontal axis is calculated using fuel rate.
- The vertical axis is the original fuel rate.

The two axes are therefore not independent. A strong relationship may exist
partly because the same measurement is present on both axes.

## Plot 4: relative altitude

The intended fourth horizontal-axis value is

$$
\Pi_4 = \frac{z}{x},
$$

where:

- $z$ is altitude; and
- $x$ is total vehicle distance.

This represents altitude relative to the distance scale.

In simple terms, the plot would ask:

> Does fuel rate change according to relative altitude?

The current program cannot generate this plot because the parameter worksheet
defines `GPS_Altitude[m]`, but the measurement worksheet does not contain a
column with that name.

## Overall interpretation

The plots are an exploratory analysis. They investigate whether fuel rate has
a simple straight-line relationship with different unit-free combinations of
vehicle measurements.

The $\Pi$ calculations make the physical quantities easier to compare, while
the fitted lines provide a quick summary of their linear relationships. The
results still need engineering interpretation, data-quality checks, and more
statistical analysis before they can support conclusions about vehicle
behavior or fuel efficiency.
