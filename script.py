import os
import glob
import pandas as pd
from itertools import combinations

# Helper function to load .dat data files from the "data" folder
def load_data(directory="data"):
    """Load all .dat files in the specified directory."""
    data = {}
    file_pattern = os.path.join(directory, "*.dat")  # Path pattern for .dat files
    for filepath in glob.glob(file_pattern):
        filename = os.path.basename(filepath)
        material = filename.split('.')[0]  # Extract material and diameter from filename
        df = pd.read_csv(filepath, delim_whitespace=True, comment='#', names=["nm", "Qext", "Qsca", "Qabs"], header=None)
        df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')  # Ensure wavelength is numeric
        data[material] = df
    return data

def calculate_absorption(data, nanoparticle, wavelength):
    """Fetch the absorption cross-section for a given nanoparticle and wavelength."""
    df = data[nanoparticle]
    target_row = df.loc[pd.to_numeric(df.iloc[:, 0], errors='coerce') == wavelength]
    if not target_row.empty:
        return target_row['Qabs'].values[0]  # 'Qabs' from .dat files
    else:
        raise ValueError(f"Wavelength {wavelength} not found for {nanoparticle}")

def calculate_scattering(data, nanoparticle, wavelength):
    """Fetch the scattering cross-section for a given nanoparticle and wavelength."""
    df = data[nanoparticle]
    target_row = df.loc[pd.to_numeric(df.iloc[:, 0], errors='coerce') == wavelength]
    if not target_row.empty:
        return target_row['Qsca'].values[0]  # 'Qsca' from .dat files
    else:
        raise ValueError(f"Wavelength {wavelength} not found for {nanoparticle}")

# Function to calculate the S score based on the formula
def calculate_S_score(n, combination, data, weights):
    total_score = 0
    for i in range(n):
        nanoparticle, wavelength = combination[i]
        
        # Calculate absorption and scattering cross-sections
        absorptionCrossSection = calculate_absorption(data, nanoparticle, wavelength)
        scatteringCrossSection = calculate_scattering(data, nanoparticle, wavelength)

        # Calculate the total cross-section score for this nanoparticle-frequency pair
        totalCrossSectionScore = (weights['scattering'] * scatteringCrossSection) + \
                                 (weights['absorption'] * absorptionCrossSection)
        
        # Sum penalties from unintended cross-talk
        total_penalty = 0
        for j in range(n):
            if i != j:
                other_nanoparticle, _ = combination[j]
                other_absorption = calculate_absorption(data, other_nanoparticle, wavelength)
                other_scattering = calculate_scattering(data, other_nanoparticle, wavelength)

                penalty_cross_section = (weights['scattering'] * other_scattering) + \
                                        (weights['absorption'] * other_absorption)
                total_penalty += penalty_cross_section
        
        # Update the score with the subtraction of the total penalty
        nanoparticle_score = totalCrossSectionScore - total_penalty
        
        # Sum the absolute value of each nanoparticle's score
        total_score += nanoparticle_score
    
    return total_score

# Main function to find the best combination based on the S score
def find_best_combination(n, nanoparticleData, weights):
    maxScore = float('-inf')
    bestCombination = None

    # Generate all possible combinations of n nanoparticle-wavelength pairings
    nanoparticles = list(nanoparticleData.keys())
    wavelengths = nanoparticleData[nanoparticles[0]].iloc[:, 0].unique()

    # Iterate through each combination of nanoparticle and wavelength pairings
    for combination in combinations([(np, wl) for np in nanoparticles for wl in wavelengths], n):
        used_nanoparticles = set()

        # Check if nanoparticle (material + size) or wavelength is reused in the combination
        valid_combination = True
        for np, wl in combination:
            if np in used_nanoparticles:
                valid_combination = False
                break
            used_nanoparticles.add(np)

        if not valid_combination:
            continue  # Skip if any nanoparticle or wavelength is reused in this combination

        # Calculate S score for valid combination
        score = calculate_S_score(n, combination, nanoparticleData, weights)

        if score > maxScore:
            maxScore = score
            bestCombination = combination

            # Print the new best combination and its S score
            print(f"New Best Combination: {bestCombination}, S-score: {maxScore}")

    return bestCombination, maxScore

# Example Usage
def main():
    # Load .dat files from the 'data' directory
    nanoparticleData = load_data()

    # Define weights for scattering and absorption
    weights = {'scattering': 0.5, 'absorption': 0.5}

    # Find the best combination of 3 nanoparticle-frequency pairings
    best_combination, max_score = find_best_combination(3, nanoparticleData, weights)

    # Final result print
    print(f"Final Best Combination: {best_combination}")
    print(f"Max Score: {max_score}")

if __name__ == "__main__":
    main()
