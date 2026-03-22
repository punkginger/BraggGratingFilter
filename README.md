# Theory

## Parameters
A standard QCL is a multi-mode Fabry-perot laser, with bragg grating curved on its surface, we create a stopband at certain frequency $f_{br}$. Moreover, if we introduce $\pi\text{-shift}$ into the bragg grating, which means moving the grating left or right for a small distance($\delta$) and enlarge the etch at the center of the grating to be 2 times of its regular length($L_{pi}$), we can create a passband inside the stopband, therefore we get a precise bandpass filter. It's tunable via tweaking the temperature or the voltage applied.

Now, supposing we are designing a THz QCL laser with Bragg Grating acting as a bandpass filter to make the laser a single-mode one. Firstly, the parameters we already have are:

1. $f_{br}$ or $\lambda_{br}$ which is the desired frequency,
2. $n_{mm}$ and $n_{ee}$ which are the real part of the refractive index of the metal and etch part of the laser.
3. $\alpha_m$ and $\alpha_e$ which are the absorption coefficient of the metal and etch part,
4. $L_c$ which is the total length of the laser chip.

Then, we'll have to calculate the length of the metal($L_m$) and etch($L_e$) part, which together form the bragg grating. The period of bragg grating($\Lambda$) on QCL, which means the length of one metal region plus one etch region, is defined as:

$$\Lambda=\frac{\lambda_{br}}{2n_{eff}}$$

Where $n_{eff}$ is the average refractive index between the two materials and it's defined as:

$$n_{eff}=(n_{mm}\sigma)+[n_{ee}(1-\sigma)]$$

Where $\sigma$ is the duty cycle of the bragg grating, which basically means the proportion of the metal part over the etch part.
Therefore, it's easy to see that there's certain connection between $L_m$ and $L_e$:

$$L_m=\Lambda\sigma \\ L_e=\Lambda(1-\sigma)$$

Moreover, we define $L_{laser}=L_c-N\Lambda$.
To push our calculation of the transfer matrices, we need to know the refractive index of the two materials. Refractive index($n$) consists of two parts:

$$ n=n_{real}-k\times i$$

And the constant $k$ in the imaginary part is defined as:

$$k=\frac{\lambda_{br}}{4\pi\alpha}$$

And the propagation const($\beta$) which is crucial to transfer matrix is defined as:

$$\beta=\frac{2\pi}{\lambda_{br}n}$$

It means how many radius the wave's phase have changed per meter it travels.
After all these, we can get the propagation const of the metal($\beta_{m}$) and etch($\beta_{e}$) part, and we can calculate the phase shift by:

$$phi_+ = \beta_m  L_m + \beta_e  L_e \\ phi_- = \beta_m  L_m - \beta_e  L_e$$

Moreover, the reflection($r$) and transmission($t$) coefficients at the interface are:

$$r=\frac{n_{ee}-n_{mm}}{n_{ee}+n_{mm}} $$
$$ t=\sqrt{1-r^2}$$

## Matrix Construct
The total transmission matrix of the QCL laser($T_{QCL}$) is defined as:
$$T_{QCL}=T_F\times T_{PL}\times T_R$$
where $T_F$ means the T-matrix of the front part of the laser, $T_{PL}$ means that of the photonic lattice part and $T_R$ means that of the rare part of the laser.

To construct the photonic lattice matrix ($T_{PL}$), we first build the transfer matrix for a single standard Bragg period ($T_{bragg}$). By utilizing the accumulated phase ($\phi_+$) and phase difference ($\phi_-$) defined above, along with the interface coefficients $r$ and $t$, the elements of the Bragg matrix are defined as:

$$T_{bragg}=\begin{bmatrix} T_{11} & T_{12} \\ T_{21} & T_{22} \end{bmatrix}$$

Where the individual matrix elements are computed as follows:

$$T_{11}=\frac{1}{t^2}[e^{i\phi_+}-r^2e^{-i\phi_-}]$$
$$T_{12}=\frac{r}{t^2}[e^{-i\phi_+}-e^{i\phi_-}]$$
$$T_{21}=\frac{r}{t^2}[e^{i\phi_+}-e^{-i\phi_-}]$$
$$T_{22}=\frac{1}{t^2}[e^{-i\phi_+}-r^2e^{i\phi_-}]$$

If a $\pi\text{-shift}$ defect is introduced into the center of the grating, it acts as an isolated region consisting entirely of the etched material with a specific length ($L_{pi}$). 
In this defect region, the metal phase components drop out, leaving the phase terms as $\phi_{+\pi} = \beta_e L_{pi}$ and $\phi_{-\pi} = \beta_e L_{pi}$. 

The $\pi\text{-shift}$ transfer matrix ($T_{\pi}$) is constructed using the exact same matrix template as $T_{bragg}$, but substituting these specific $\pi\text{-shift}$ phase terms.

Next, we must account for the propagation of the wave through the remaining unstructured laser cavity ($L_{laser}$), which is effectively split into the front and rear sections of the chip. Incorporating our physical tuning parameter $\delta$ (the spatial deviation of the grating from the center), the propagation phases for the front ($\phi_F$) and rear ($\phi_R$) extremities are:

$$\phi_F=\beta_m\left(\frac{L_{laser}-L_{pi}}{2}-\delta\right)$$
$$\phi_R=\beta_m\left(\frac{L_{laser}-L_{pi}}{2}+\delta\right)$$

These phase shifts yield simple diagonal propagation matrices for the outer cavities:

$$T_F=\begin{bmatrix} e^{i\phi_F} & 0 \\ 0 & e^{-i\phi_F} \end{bmatrix}, \quad T_R=\begin{bmatrix} e^{i\phi_R} & 0 \\ 0 & e^{-i\phi_R} \end{bmatrix}$$

With all regional matrices defined, the global transfer matrix of the entire QCL device ($T_{QCL}$) is calculated via sequential matrix multiplication. For a standard periodic grating (without a defect), the photonic lattice is simply the Bragg matrix multiplied by the number of periods $N$:

$$T_{QCL}=T_F \times (T_{bragg})^N \times T_R$$

For a defect-engineered filter with a $\pi\text{-shift}$, the Bragg periods are split evenly across the central defect:

$$T_{QCL}=T_F \times (T_{bragg})^{N/2} \times T_{\pi} \times (T_{bragg})^{N/2} \times T_R$$

Finally, to extract measurable physical quantities such as reflected and transmitted power, we convert our global Transfer Matrix into a Scattering Matrix (S-Matrix). The transfer matrix relates the forward ($E^+$) and backward ($E^-$) traveling electric fields on the left side of the device to those on the right side:

$$\begin{pmatrix} E_{left}^+ \\ E_{left}^- \end{pmatrix} = \begin{pmatrix} T_{11} & T_{12} \\ T_{21} & T_{22} \end{pmatrix} \begin{pmatrix} E_{right}^+ \\ E_{right}^- \end{pmatrix}$$

Expanding this matrix multiplication yields two fundamental field equations:

$$E_{left}^+ = T_{11} E_{right}^+ + T_{12} E_{right}^-$$
$$E_{left}^- = T_{21} E_{right}^+ + T_{22} E_{right}^-$$

To find the transmission and reflection properties when light is shining strictly from left to right, we assume there is no input from the right side ($E_{right}^- = 0$). Substituting this into our expanded equations gives:

$$E_{left}^+ = T_{11} E_{right}^+$$
$$E_{left}^- = T_{21} E_{right}^+$$

From these, we can define the forward transmission ($S_{21}$) and reflection ($S_{11}$) coefficients:

$$S_{21} = \frac{E_{right}^+}{E_{left}^+} = \frac{1}{T_{11}}$$
$$S_{11} = \frac{E_{left}^-}{E_{left}^+} = \frac{T_{21}}{T_{11}}$$

Conversely, to find the transmission from the right side, we assume light is only shining from right to left, meaning there is no input from the left side ($E_{left}^+ = 0$). Setting $E_{left}^+ = 0$ in our very first equation gives:

$$0 = T_{11} E_{right}^+ + T_{12} E_{right}^-$$
$$E_{right}^+ = -\frac{T_{12}}{T_{11}} E_{right}^-$$

Substituting this expression for $E_{right}^+$ back into the equation for $E_{left}^-$ yields:

$$E_{left}^- = T_{21} \left( -\frac{T_{12}}{T_{11}} E_{right}^- \right) + T_{22} E_{right}^-$$
$$E_{left}^- = \left( \frac{-T_{12}T_{21} + T_{11}T_{22}}{T_{11}} \right) E_{right}^-$$

Recognizing that the numerator term $(T_{11}T_{22} - T_{12}T_{21})$ is simply the determinant of the global transfer matrix ($\det(T_{QCL})$), we can define the backward transmission coefficient ($S_{12}$):

$$S_{12} = \frac{E_{left}^-}{E_{right}^-} = \frac{\det(T_{QCL})}{T_{11}}$$

And for a laser with reciprocal structure like our QCL, $\det(T_{QCL})=1$, thus $S_{12}=S_{21}$

Because optical power is proportional to the square of the amplitude, the final measurable Reflected Power ($R$) and Transmitted Power ($T$) spectra across our evaluated frequencies are:

$$R=|S_{11}|^2$$
$$T=|S_{12}|^2$$

By evaluating these equations over a target frequency array, we can pinpoint the exact frequency of the stopband.

## parameter optimization
Now, back to our designment progress, I judge the quality of the design based on the transmission peak. To achieve the highest possible transmission at our target resonance, we must optimize the physical structure of the grating. However, not all variables in our mathematical model can be freely tuned during the design phase.

1. The refractive indices ($n_{mm}$, $n_{ee}$) and absorption coefficients ($\alpha_m$, $\alpha_e$) are intrinsic properties of the chosen semiconductor materials and metals. Once the material system is selected, these are completely fixed.
2. The total chip length ($L_c$) is determined during the cleaving process of the laser cavity and remains constant.
3. The $\pi\text{-shift}$ defect length ($L_{pi}$) is strictly defined as exactly twice the length of a standard etched section ($2L_e$) to achieve the perfect phase inversion.


As for the parameter $\delta$, one might assume that shifting the entire grating slightly off-center by a spatial deviation $\delta$ would alter the interference pattern and tune the filter. However, evaluating the Transfer Matrix reveals that for a device without facet reflections, $\delta$ has absolutely no impact on the final output power.

To prove this, let us group the entire central grating structure (both the Bragg periods and the $\pi\text{-shift}$) into a single core matrix, $T_{core}$:

$$T_{core} = \begin{bmatrix} A & B \\ C & D \end{bmatrix}$$

The total system matrix is defined by sandwiching this core between the front and rear propagation matrices: $T_{QCL} = T_F \times T_{core} \times T_R$. 

Let us define a base phase, $\phi_0$, which represents exactly half of the unstructured cavity length:

$$\phi_0 = \beta_m\left(\frac{L_{laser}-L_{pi}}{2}\right)$$

According to our previous definitions, the front and rear phases incorporating the spatial shift $\delta$ are:

$$\phi_F = \phi_0 - \beta_m\delta$$
$$\phi_R = \phi_0 + \beta_m\delta$$

When we perform the matrix multiplication $T_F \times T_{core} \times T_R$, the elements of the resulting total matrix $T_{QCL}$ combine the phase exponents. For addition, the $\delta$ terms perfectly cancel out: $\phi_F + \phi_R = 2\phi_0$. For subtraction, they compound: $\phi_F - \phi_R = -2\beta_m\delta$. This yields the following total matrix:

$$T_{QCL} = \begin{bmatrix} A e^{i2\phi_0} & B e^{-i2\beta_m\delta} \\ C e^{i2\beta_m\delta} & D e^{-i2\phi_0} \end{bmatrix}$$

While the $\delta$ parameter survives in the off-diagonal elements ($T_{12}$ and $T_{21}$), it completely vanishes from the primary diagonal ($T_{11}$ and $T_{22}$). Because optical sensors measure power (the absolute squared magnitude) rather than complex amplitude, we apply our S-matrix power equations:

$$T_{power} = |S_{12}|^2 = \left| \frac{\det(T_{QCL})}{A e^{i2\phi_0}} \right|^2$$
$$R = |S_{11}|^2 = \left| \frac{C e^{i2\beta_m\delta}}{A e^{i2\phi_0}} \right|^2$$

A fundamental property of complex exponentials is that their absolute magnitude is always exactly $1$ (e.g., $|e^{ix}|^2 = 1$). Therefore, the phase terms mathematically collapse, simplifying the power equations to:

$$T_{power} = \frac{|\det(T_{QCL})|^2}{|A|^2}$$
$$R = \frac{|C|^2}{|A|^2}$$

Consequently, $\delta$ dictates the phase of the output wave but leaves the measurable optical power completely unchanged.

Therefore, the only parameters left are $L_m$, $L_e$ and $N$ and the former two are basically one thing since they have certain connection between each other and can be assessed using one single parameter: duty cycle $\sigma$

Moreover, I introduce 3 factors into assessing the quality of our laser:

1. Peak Transmission Power ($T_{peak}$)The absolute maximum amplitude of the defect mode spike. Overcoming material absorption to maintain a viable signal output is paramount.
2. Quality Factor ($Q$)This defines the sharpness and precision of the bandpass filter. It is calculated by dividing the central target frequency by the Full Width at Half Maximum (FWHM) of the transmission peak:
   
$$Q = \frac{f_{br}}{\Delta f_{FWHM}}$$

3. Stopband Rejection RatioThis measures the contrast between the desired signal (the defect peak) and the noise floor of the filter (the stopband). However, measuring this algorithmically presents a physical challenge: due to the finite nature of the cavity and the heavy material losses, the outer edges of the stopband are heavily distorted by Fabry-Perot ripples. Blindly searching for valleys can result in measuring these ripples instead of the true noise floor.To bypass this, we utilize the analytical physics of a Bragg grating to construct a "search window". The theoretical width of the stopband ($\Delta f_{sb}$) for an infinite grating is defined as:

$$\Delta f_{sb} = f_{br} \left( \frac{2|n_{mm} - n_{ee}|}{\pi n_{eff}} \right) \sin(\pi \sigma)$$

By calculating this theoretical width dynamically for every duty cycle, we establish strict frequency boundaries (a "fence") around our target resonance:

$$f_{left} = f_{br} - \frac{\Delta f_{sb}}{2} \\ f_{right} = f_{br} + \frac{\Delta f_{sb}}{2}$$

We then isolate the transmission data strictly within these boundaries and locate the absolute minimum transmission value ($T_{floor}$). This guarantees we are measuring the true darkness of the stopband, completely ignoring the surrounding Fabry-Perot ripples. The rejection ratio is then calculated as:

$$\text{Rejection Ratio} = \frac{T_{peak}}{T_{floor}}$$

Later, I'll assign each factor a weight, determine how much influence they ultimately have on the result.