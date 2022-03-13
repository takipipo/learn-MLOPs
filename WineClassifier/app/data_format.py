from pydantic import BaseModel, conlist
from typing import List

# Represents a particular wine datapoint
class Wine(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

class BatchWine(BaseModel):
    batch_of_wine: List[conlist(item_type=float, min_items=13, max_items=13)]