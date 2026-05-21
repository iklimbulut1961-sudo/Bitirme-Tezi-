# # 2. Ortam Hazırlığı ve Kurulum
# 
# Tez Google Colab (T4 GPU) ortamında çalışacak şekilde tasarlanmıştır.  

# Kütüphanelerin yüklenmesi
!pip install linearmodels --quiet
!pip install transformers torch --quiet

import os
import sys
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns

warnings.filterwarnings('ignore')

print(f"  pandas  : {pd.__version__}")
print(f"  numpy   : {np.__version__}")

