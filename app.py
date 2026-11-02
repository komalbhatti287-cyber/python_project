# python_project
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
 
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
 
# Columns used in the project
NUM_COLS = ["age", "systolic_bp", "diastolic_bp", "cholesterol",
            "glucose", "heart_rate", "bmi"]
LABELS = {
    "age": "Age (years)", "systolic_bp": "Systolic BP (mmHg)",
    "diastolic_bp": "Diastolic BP (mmHg)", "cholesterol": "Cholesterol (mg/dL)",
    "glucose": "Fasting Glucose (mg/dL)", "heart_rate": "Heart Rate (bpm)",
    "bmi": "BMI (kg/m²)",
}
# Plausible physiological ranges, used to remove bad records
VALID_RANGE = {
    "age": (0, 110), "systolic_bp": (70, 250), "diastolic_bp": (40, 150),
    "cholesterol": (80, 500), "glucose": (40, 500),
    "heart_rate": (30, 220), "bmi": (10, 70),
}
 
 
# ----------------------------------------------------------------------
# 1. DATA COLLECTION
# ----------------------------------------------------------------------
def generate_data(n=500, seed=42):
    """Create realistic synthetic patient records (with a few errors on purpose)."""
    rng = np.random.default_rng(seed)
 
    age = np.clip(rng.normal(48, 17, n), 18, 90).round()
    gender = rng.choice(["Male", "Female"], n)
    male = (gender == "Male").astype(float)
 
    bmi = np.clip(rng.normal(26, 4.5, n) + 0.02 * (age - 48), 15, 50)
    systolic = 100 + 0.50 * age + 0.8 * (bmi - 25) + 3 * male + rng.normal(0, 9, n)
    diastolic = 60 + 0.20 * age + 0.4 * (bmi - 25) + rng.normal(0, 6, n)
    chol = 150 + 0.9 * age + 1.5 * (bmi - 25) + rng.normal(0, 25, n)
    glucose = 80 + 0.25 * age + 1.2 * (bmi - 25) + rng.normal(0, 12, n)
    hr = np.clip(rng.normal(74, 9, n) - 0.05 * (age - 48), 45, 130)
 
    data = {
        "patient_id": np.arange(1001, 1001 + n),
        "gender": gender,
        "age": age, "systolic_bp": systolic.round(), "diastolic_bp": diastolic.round(),
        "cholesterol": chol.round(), "glucose": glucose.round(),
        "heart_rate": hr.round(), "bmi": bmi.round(1),
    }
 
    # Inject dirty data so the cleaning step has something to do
    for col in ["systolic_bp", "cholesterol", "glucose", "bmi"]:
        idx = rng.choice(n, 8, replace=False)
        data[col] = data[col].astype(float)
        data[col][idx] = np.nan                      # missing values
    bad = rng.choice(n, 5, replace=False)
    data["systolic_bp"][bad] = 999                   # impossible values
    data["age"][rng.choice(n, 3, replace=False)] = -5
    return data
 
 
def load_csv(path):
    """Load a CSV with columns: patient_id,gender,age,systolic_bp,diastolic_bp,
    cholesterol,glucose,heart_rate,bmi  (any numeric column may be missing)."""
    raw = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    names = raw.dtype.names
    data = {}
    n = len(raw)
    data["patient_id"] = raw["patient_id"] if "patient_id" in names else np.arange(n)
    data["gender"] = (np.array([str(g).strip().capitalize() for g in raw["gender"]])
                      if "gender" in names else np.array(["Unknown"] * n))
    for col in NUM_COLS:
        data[col] = raw[col].astype(float) if col in names else np.full(n, np.nan)
    return data
 
 
# ----------------------------------------------------------------------
# 2. CLEANING + FILTERING  (NumPy)
# ----------------------------------------------------------------------
def clean_data(data):
    """Remove rows with NaN values or physiologically impossible values."""
    n = len(data["age"])
    keep = np.ones(n, dtype=bool)
    for col in NUM_COLS:
        values = data[col].astype(float)
        lo, hi = VALID_RANGE[col]
        keep &= ~np.isnan(values) & (values >= lo) & (values <= hi)
    cleaned = {k: v[keep] for k, v in data.items()}
    cleaned["gender"] = cleaned["gender"].astype(str)
    return cleaned, n - keep.sum()
 
 
def filter_data(data, age_min, age_max, gender, only_hypertensive):
    """Boolean-mask filtering."""
    mask = (data["age"] >= age_min) & (data["age"] <= age_max)
    if gender != "All":
        mask &= data["gender"] == gender
    if only_hypertensive:
        mask &= (data["systolic_bp"] >= 140) | (data["diastolic_bp"] >= 90)
    return {k: v[mask] for k, v in data.items()}
 
 
def summary_text(data, removed=None):
    n = len(data["age"])
    if n == 0:
        return "No records match the current filter."
    lines = [f"Records analysed : {n}"]
    if removed is not None:
        lines.append(f"Dirty rows removed: {removed}")
    male = np.sum(data["gender"] == "Male")
    female = np.sum(data["gender"] == "Female")
    lines.append(f"Male / Female    : {male} / {female}\n")
    lines.append(f"{'Variable':<22}{'Mean':>8}{'Std':>8}{'Min':>8}{'Max':>8}")
    for c in NUM_COLS:
        v = data[c]
        lines.append(f"{LABELS[c]:<22.22}{v.mean():>8.1f}{v.std():>8.1f}"
                     f"{v.min():>8.1f}{v.max():>8.1f}")
    hyper = np.sum((data["systolic_bp"] >= 140) | (data["diastolic_bp"] >= 90))
    high_chol = np.sum(data["cholesterol"] >= 240)
    diabetic = np.sum(data["glucose"] >= 126)
    lines.append(f"\nHypertension (>=140/90): {hyper} ({100*hyper/n:.1f}%)")
    lines.append(f"High cholesterol (>=240): {high_chol} ({100*high_chol/n:.1f}%)")
    lines.append(f"High glucose (>=126)    : {diabetic} ({100*diabetic/n:.1f}%)")
    return "\n".join(lines)
 
 
# ----------------------------------------------------------------------
# 3. PLOTS  (each function fills a matplotlib Figure)
# ----------------------------------------------------------------------
def plot_age_hist(fig, d):
    ax = fig.add_subplot(111)
    ax.hist(d["age"], bins=15, color="#3b82f6", edgecolor="white")
    ax.axvline(d["age"].mean(), color="red", ls="--", label=f"Mean = {d['age'].mean():.1f}")
    ax.set(title="Age Distribution of Patients", xlabel="Age (years)", ylabel="Patients")
    ax.legend()
 
 
def plot_gender_bar(fig, d):
    ax = fig.add_subplot(111)
    genders, counts = np.unique(d["gender"], return_counts=True)
    bars = ax.bar(genders, counts, color=["#ec4899", "#3b82f6", "#9ca3af"][:len(genders)])
    ax.bar_label(bars)
    ax.set(title="Patients by Gender", ylabel="Patients")
 
 
def plot_age_group_bar(fig, d):
    ax = fig.add_subplot(111)
    edges = [0, 30, 40, 50, 60, 70, 120]
    names = ["<30", "30-39", "40-49", "50-59", "60-69", "70+"]
    group = np.digitize(d["age"], edges[1:-1])
    x = np.arange(len(names))
    w = 0.38
    for i, (g, color) in enumerate([("Male", "#3b82f6"), ("Female", "#ec4899")]):
        counts = np.bincount(group[d["gender"] == g], minlength=len(names))
        bars = ax.bar(x + (i - 0.5) * w, counts, w, label=g, color=color)
        ax.bar_label(bars, fontsize=8)
    ax.set_xticks(x, names)
    ax.set(title="Age Groups by Gender", xlabel="Age group", ylabel="Patients")
    ax.legend()
 
 
def plot_vitals_hist(fig, d):
    cols = ["systolic_bp", "cholesterol", "glucose", "bmi"]
    colors = ["#ef4444", "#f59e0b", "#10b981", "#8b5cf6"]
    for i, (c, col) in enumerate(zip(cols, colors), 1):
        ax = fig.add_subplot(2, 2, i)
        ax.hist(d[c], bins=15, color=col, edgecolor="white")
        ax.set(title=LABELS[c], ylabel="Patients")
    fig.tight_layout()
 
 
def _scatter(fig, d, xc, yc, color_by_gender=True):
    ax = fig.add_subplot(111)
    if color_by_gender:
        for g, col in [("Male", "#3b82f6"), ("Female", "#ec4899")]:
            m = d["gender"] == g
            ax.scatter(d[xc][m], d[yc][m], s=18, alpha=0.6, c=col, label=g)
        ax.legend()
    else:
        ax.scatter(d[xc], d[yc], s=18, alpha=0.6)
    # Least-squares trend line using NumPy
    if len(d[xc]) > 2:
        slope, intercept = np.polyfit(d[xc], d[yc], 1)
        xs = np.linspace(d[xc].min(), d[xc].max(), 100)
        r = np.corrcoef(d[xc], d[yc])[0, 1]
        ax.plot(xs, slope * xs + intercept, "k--", lw=1.5, label=f"r = {r:.2f}")
        ax.legend()
    ax.set(title=f"{LABELS[yc]} vs {LABELS[xc]}", xlabel=LABELS[xc], ylabel=LABELS[yc])
    ax.grid(alpha=0.3)
 
 
def plot_age_bp(fig, d):
    _scatter(fig, d, "age", "systolic_bp")
 
 
def plot_age_chol(fig, d):
    _scatter(fig, d, "age", "cholesterol")
 
 
def plot_bp_chol(fig, d):
    _scatter(fig, d, "systolic_bp", "cholesterol")
 
 
def plot_bmi_glucose(fig, d):
    _scatter(fig, d, "bmi", "glucose")
 
 
def plot_heatmap(fig, d):
    ax = fig.add_subplot(111)
    matrix = np.corrcoef(np.vstack([d[c] for c in NUM_COLS]))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    short = ["Age", "SysBP", "DiaBP", "Chol", "Gluc", "HR", "BMI"]
    ax.set_xticks(range(len(short)), short)
    ax.set_yticks(range(len(short)), short)
    for i in range(len(short)):
        for j in range(len(short)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Correlation (r)")
    ax.set_title("Correlation Matrix")
 
 
def plot_boxplot(fig, d):
    ax = fig.add_subplot(111)
    groups = [d["systolic_bp"][d["gender"] == g] for g in ("Male", "Female")]
    ax.boxplot(groups, tick_labels=["Male", "Female"], patch_artist=True)
    ax.set(title="Systolic BP by Gender", ylabel="Systolic BP (mmHg)")
    ax.grid(alpha=0.3)
 
 
# ----------------------------------------------------------------------
# 4. GUI  – main window + a new window for every chart
# ----------------------------------------------------------------------
class HealthApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Health Data Analysis – Control Panel")
        self.geometry("720x640")
 
        self.raw = generate_data()
        self.clean, self.removed = clean_data(self.raw)
        self.view = self.clean            # currently filtered data
 
        self._build_ui()
        self.apply_filter()
 
    # ---- layout ----
    def _build_ui(self):
        top = ttk.LabelFrame(self, text="1. Data source")
        top.pack(fill="x", padx=10, pady=6)
        ttk.Button(top, text="Generate new sample data", command=self.new_sample).pack(side="left", padx=5, pady=5)
        ttk.Button(top, text="Load CSV file…", command=self.load_file).pack(side="left", padx=5)
 
        flt = ttk.LabelFrame(self, text="2. Filter (NumPy boolean masks)")
        flt.pack(fill="x", padx=10, pady=6)
        ttk.Label(flt, text="Age from").grid(row=0, column=0, padx=4, pady=5)
        self.age_min = tk.StringVar(value="18")
        ttk.Entry(flt, textvariable=self.age_min, width=5).grid(row=0, column=1)
        ttk.Label(flt, text="to").grid(row=0, column=2, padx=4)
        self.age_max = tk.StringVar(value="90")
        ttk.Entry(flt, textvariable=self.age_max, width=5).grid(row=0, column=3)
        ttk.Label(flt, text="Gender").grid(row=0, column=4, padx=8)
        self.gender = tk.StringVar(value="All")
        ttk.Combobox(flt, textvariable=self.gender, values=["All", "Male", "Female"],
                     width=8, state="readonly").grid(row=0, column=5)
        self.hyper = tk.BooleanVar()
        ttk.Checkbutton(flt, text="Only hypertensive", variable=self.hyper).grid(row=0, column=6, padx=10)
        ttk.Button(flt, text="Apply", command=self.apply_filter).grid(row=0, column=7, padx=5)
 
        plots = ttk.LabelFrame(self, text="3. Visualise  (each opens in a new window)")
        plots.pack(fill="x", padx=10, pady=6)
        buttons = [
            ("Age histogram", plot_age_hist),
            ("Gender bar plot", plot_gender_bar),
            ("Age groups by gender", plot_age_group_bar),
            ("Vitals histograms", plot_vitals_hist),
            ("Age vs Blood Pressure", plot_age_bp),
            ("Age vs Cholesterol", plot_age_chol),
            ("BP vs Cholesterol", plot_bp_chol),
            ("BMI vs Glucose", plot_bmi_glucose),
            ("Correlation heatmap", plot_heatmap),
            ("BP box plot", plot_boxplot),
        ]
        for i, (text, fn) in enumerate(buttons):
            ttk.Button(plots, text=text, width=24,
                       command=lambda f=fn, t=text: self.open_plot(t, f)
                       ).grid(row=i // 3, column=i % 3, padx=5, pady=4)
 
        out = ttk.LabelFrame(self, text="Summary statistics")
        out.pack(fill="both", expand=True, padx=10, pady=6)
        self.text = tk.Text(out, font=("Consolas", 10), height=16)
        self.text.pack(fill="both", expand=True, padx=4, pady=4)
 
    # ---- actions ----
    def new_sample(self):
        seed = int(np.random.randint(0, 10_000))
        self.raw = generate_data(seed=seed)
        self.clean, self.removed = clean_data(self.raw)
        self.apply_filter()
 
    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        try:
            self.raw = load_csv(path)
            self.clean, self.removed = clean_data(self.raw)
        except Exception as e:
            messagebox.showerror("Error", f"Could not read file:\n{e}")
            return
        self.apply_filter()
 
    def apply_filter(self):
        try:
            lo, hi = float(self.age_min.get()), float(self.age_max.get())
        except ValueError:
            messagebox.showwarning("Invalid input", "Age limits must be numbers.")
            return
        self.view = filter_data(self.clean, lo, hi, self.gender.get(), self.hyper.get())
        self.text.delete("1.0", "end")
        self.text.insert("end", summary_text(self.view, self.removed))
 
    def open_plot(self, title, plot_fn):
        if len(self.view["age"]) < 3:
            messagebox.showinfo("Not enough data", "Too few records – relax the filter.")
            return
        win = tk.Toplevel(self)                    # <- the NEW window
        win.title(title)
        win.geometry("820x600")
        fig = Figure(figsize=(8, 5.5), dpi=100)
        plot_fn(fig, self.view)
        canvas = FigureCanvasTkAgg(fig, master=win)
        NavigationToolbar2Tk(canvas, win).update()  # zoom / pan / save
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()
 
 
if __name__ == "__main__":
    HealthApp().mainloop()
