"""Project-wide configuration for TEC surrogate modeling."""

from pathlib import Path

# === Paths ===
ROOT = Path(r"D:\world model")
PROJECT = ROOT / "tec_surrogate"
DATA = PROJECT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
MODELS_DIR = DATA / "models"
MODELS_MPH = PROJECT / "models"
OUTPUTS = PROJECT / "outputs"
FIGURES = OUTPUTS / "figures"

SOURCE_MODEL = ROOT / r"thermoelectric_cooler.zh_CN (1).mph"
SIMPLIFIED_MODEL = MODELS_MPH / "tec_1pair.mph"
WORKING_MODEL = MODELS_MPH / "tec_1pair_working.mph"

# === Parameter metadata ===
# (name, comsol_key, unit, nominal, lower, upper)
PARAMETERS = [
    # 1-pair TEC. I0 is NOT independent — controlled by study 3 parametric sweep (0.1-5A, 20 steps).
    ("length",       "length",       "mm",  4.0,   2.0,  7.0),
    ("width",        "width",        "mm",  3.0,   1.5,  5.0),
    ("height",       "height",       "mm",  2.5,   1.25, 3.75),
    ("d_conductor",  "d_conductor",  "um",  100,   50,   150),
    ("d_ceramics",   "d_ceramics",   "mm",  0.3,   0.15, 0.45),
    ("leg_length",   "leg_length",   "mm",  1.0,   0.5,  1.5),
    ("leg_width",    "leg_width",    "mm",  1.2,   0.6,  1.8),
    ("pitch",        "pitch",        "mm",  0.5,   0.25, 0.75),
    ("Tref",         "Tref",         "K",   323.15, 300,  350),
    ("dT0",          "dT0",          "K",   50,    10,   80),
]
# I0 is NOT in PARAMETERS — it's the parametric sweep variable in study 3
N_PARAMS = len(PARAMETERS)  # 10

PARAM_NAMES = [p[0] for p in PARAMETERS]
PARAM_COMSOL_KEYS = [p[1] for p in PARAMETERS]
PARAM_UNITS = [p[2] for p in PARAMETERS]
PARAM_NOMINAL = [p[3] for p in PARAMETERS]
PARAM_LOWER = [p[4] for p in PARAMETERS]
PARAM_UPPER = [p[5] for p in PARAMETERS]
N_PARAMS = len(PARAMETERS)

# === Geometric feasibility constraints ===
# Operates on PHYSICAL units: [length, width, height, d_conductor, d_ceramics,
#                              leg_length, leg_width, pitch, Tref, dT0, I0]
def check_geometric_feasibility(params_phys):
    """Return True if geometry parameters are physically feasible for 1-pair TEC.

    With n_length=2, n_width=1 fixed:
      network_length = 2*leg_length + pitch
      network_width  = leg_width
    Substrate (length/width) must contain network with reasonable margin.
    """
    L, W, H, d_cond, d_cer, ll, lw, pitch = params_phys[:8]
    d_cond_mm = d_cond / 1000  # um → mm

    # 1. Layer thickness consistency
    if H <= 2 * (d_cond_mm + d_cer):
        return False
    leg_h = H - 2 * (d_cond_mm + d_cer)
    if leg_h <= 0:
        return False

    # 2. Network dimensions (1 pair: n_length=2, n_width=1)
    net_len = 2 * ll + pitch
    net_wid = lw

    # 3. Substrate must contain the network
    if L < net_len:
        return False
    if W < net_wid:
        return False

    # 4. Pitch must be positive and less than leg dimensions (spacing between legs)
    #    For the Array feature to work: pitch < leg_length for length direction
    if pitch <= 0:
        return False
    if pitch >= ll:  # pitch must fit between legs along length
        return False

    # 5. Basic sanity: leg area must be less than substrate area
    if net_len * net_wid > L * W:
        return False

    return True


# === Region definitions ===
REGIONS = {
    "cold_ceramic":   {"id": 0, "type": "bulk", "grid": (4, 4, 4)},
    "cold_conductor": {"id": 1, "type": "bulk", "grid": (4, 4, 3)},
    "p_leg":          {"id": 2, "type": "leg",  "grid": (8, 8, 12)},
    "n_leg":          {"id": 3, "type": "leg",  "grid": (8, 8, 12)},
    "hot_conductor":  {"id": 4, "type": "bulk", "grid": (4, 4, 3)},
    "hot_ceramic":    {"id": 5, "type": "bulk", "grid": (4, 4, 4)},
}

N_REGION_TYPES = len(REGIONS)

# Component roles — stable semantic labels (not dependent on COMSOL domain numbering)
# Each role gets a unique index for one-hot encoding
COMPONENT_ROLES = [
    "cold_ceramic_slab",
    "cold_conductor_left",
    "cold_conductor_right",
    "p_leg",
    "n_leg",
    "hot_conductor_left",
    "hot_conductor_right",
    "hot_ceramic_slab",
]

N_COMPONENT_ROLES = len(COMPONENT_ROLES)

# Map region_type → component_role(s)
REGION_TO_COMPONENT = {
    "cold_ceramic":   ["cold_ceramic_slab"],
    "cold_conductor": ["cold_conductor_left", "cold_conductor_right"],
    "p_leg":          ["p_leg"],
    "n_leg":          ["n_leg"],
    "hot_conductor":  ["hot_conductor_left", "hot_conductor_right"],
    "hot_ceramic":    ["hot_ceramic_slab"],
}

# === Output expressions (to be verified by 02_probe_expressions.py) ===
SCALAR_EXPRESSIONS = {
    "delta_T_K":       {"expr": "aveop1(T)-aveop2(T)",                "unit": "K"},
    "cooling_power_W": {"expr": "abs(ht.ntefluxInt)",                  "unit": "W"},
    # ec.V0_2/ec.I0_2 for study 3; ec.V0_1/ec.I0_1 for study 2/global
    "input_power_W":   {"expr": "abs(ec.V0_2*ec.I0_2)",               "unit": "W"},
    "terminal_voltage_V": {"expr": "ec.V0_2",                          "unit": "V"},
    "cold_avg_K":      {"expr": "aveop1(T)",                           "unit": "K"},
    "hot_avg_K":       {"expr": "aveop2(T)",                           "unit": "K"},
    "equivalent_resistance_ohm": {"expr": "abs(ec.V0_2/(ec.I0_2+1e-9[A]))", "unit": "ohm"},
}

FIELD_EXPRESSIONS = {
    "T":  "T",
    "V":  "V",
    "E":  "ec.normE",
}

TIME_CURVE_EXPRESSIONS = {
    "delta_T":       "aveop1(T)-aveop2(T)",
    "cooling_power": "abs(ht.ntefluxInt)",
    "input_power":   "abs(ec.V0_2*ec.I0_2)",
    "voltage":       "ec.V0_2",
}

# Derived quantities (not separately modeled):
#   COP = Qc / Pin  (masked when Pin ≈ 0)
#   R_eq = V / I    (consistency check)

# === Time domain ===
N_TIME_POINTS = 64  # fixed sampling for curves

# === Spatial probes ===
INTERFACE_DELTA_FACTOR = 0.01  # offset as fraction of local layer thickness

# === Study names in simplified model ===
STUDY_STATIONARY = None   # to be discovered
STUDY_TRANSIENT = None    # to be discovered / created
DATASET_STATIONARY = None
DATASET_TRANSIENT = None

# === Sobol design ===
SOBOL_SEED = 42
SOBOL_M = 8  # 256 base
N_TRAIN = 160
N_CALIBRATION = 32
N_VAL = 32
N_TEST = 32

# === DeepONet hyperparameters ===
LATENT_DIM = 48
N_TIME_QUANTITIES = 4
N_SPACE_QUANTITIES = 3
N_SCALARS = 7
N_ENSEMBLE = 5

# === Training ===
BATCH_SIZE = 256
MAX_EPOCHS = 1000
EARLY_STOP_PATIENCE = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.1
