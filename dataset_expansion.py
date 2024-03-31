import numpy as np
import pandas as pd

from tqdm.notebook import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors, PandasTools, AllChem, MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors
from mordred import Calculator, descriptors


def add_molecules_from_smiles(input_df):
    """
    Add a 'molecules' col containing molecule structures using the 'SMILES' col from input_df

    Args:
        input_df (pd df): input df that contains SMILES in the 'SMILES' col

    Returns:
        0 (int): indicates the normal running of the function
    """
    PandasTools.AddMoleculeColumnToFrame(
        # Add molecular structures into input_df using SMILES in 'SMILES'
        # col and put all molecules structures into the 'molecules' col
        input_df,
        smilesCol='SMILES',
        molCol='molecules'
    )

    return 0


def generate_rdkit_descriptor_df(input_df):
    """
    Generate a pd df containing only RDKit descriptors using molecule
    structures from input_df

    Args:
        input_df (pd df): input df that contains molecule structure in the
        'molecules' col

    Returns:
        output_df (pd df): output df that contains only the generated
        RDKit descriptors

    """
    descriptor_names = [descriptor[0] for descriptor in Descriptors._descList]

    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(
        descriptor_names)

    descriptors_list = []
    for molecule in tqdm(
            input_df['molecules'],
            desc='Generating {} RDKit descriptors'.format(
                len(descriptor_names))
    ):
        try:
            descriptors = calculator.CalcDescriptors(molecule)
            descriptors_list.append(descriptors)
        except Exception as e:
            print('Error encountered when calculating RDKit descriptors for '
                  'molecule {}'.format(molecule))

    output_df = pd.DataFrame(
        descriptors_list,
        columns=descriptor_names  # Use the original descriptor_names as col
        # names
    )
    return output_df


def generate_morgan_fingerprint_df(input_df):
    """
    Generate a pd df containing only Morgan fingerprints using molecule
    structures from input_df

    Args:
        input_df (pd df): input df that contains molecule structure in the
        'molecules' col

    Returns:
        output_df (pd df): output df that contains only the generated
        Morgan fingerprints
    """
    morgan_list = [
        AllChem.GetMorganFingerprintAsBitVect(
            molecule,
            3,  # This is radius
            nBits=4096,
            # Used a rather big number here (4096) of bits to avoid bit
            # clashing
            useFeatures=True
        ).ToBitString() for molecule in tqdm(
            input_df['molecules'],
            desc='Generating Morgan fingerprints'
        )
    ]

    morgan_np = np.array(
        [
            list(bit) for bit in morgan_list
        ],
        dtype='int'
    )
    output_df = pd.DataFrame(morgan_np)
    return output_df


def generate_maccs_key_df(input_df):
    """
    Generate a pd df containing only MACCS keys using molecule
    structures from input_df

    Args:
        input_df (pd df): input df that contains molecule structure in the
        'molecules' col

    Returns:
        output_df (pd df): output df that contains only the generated
        Morgan fingerprints
    """
    maccs_list = [
        MACCSkeys.GenMACCSKeys(molecule) for molecule in
        tqdm(
            input_df['molecules'],
            desc='Generating MACCS keys',
        )
    ]

    maccs_np = np.array(
        [
            list(bit_vect) for bit_vect in maccs_list
        ],
        dtype='int'
    )
    output_df = pd.DataFrame(maccs_np)

    return output_df


def merge_multiple_dfs(df_list, left_index_bool=True, right_index_bool=True):
    """
    Merge multiple df from a df_list into a single df

    Args:
        df_list (list): list of DataFrames to be merged.
        left_index_bool (bool): use the index from the left DataFrame as the
        join key(s). Defaults to True.
        right_index_bool (bool): use the index from the right DataFrame as the
        join key(s). Defaults to True.

    Returns:
        merged_df (pd df): the output merged df.
    """

    merged_df = df_list[0]  # Use the 1st df as the base for merged_df

    for df in df_list[1:]:  # Keep merging the rest df into the merged_df
        merged_df = pd.merge(
            merged_df,
            df,
            left_index=left_index_bool,
            right_index=right_index_bool
        )

    return merged_df


def dataset_feature_expansion(input_df):
    """
    Expand input_df by adding mordred descriptors, Morgan
    fingerprints, and MACCS keys

    Args:
        input_df (pd df): input df that contains SMILES in the 'SMILES' col

    Returns:
        output_df (pd df): expanded df that contains the additional
        fingerprints and descriptors
    """
    if add_molecules_from_smiles(input_df) != 0:
        print('Something is wrong with adding molecule structures!')
    else:  # Successfully added 'molecules' col to input_df
        descriptor_df = generate_mordred_descriptor_df(input_df)
        morgan_fingerprint_df = generate_morgan_fingerprint_df(input_df)
        maccs_key_df = generate_maccs_key_df(input_df)

        merged_df = merge_multiple_dfs(
            [
                input_df,
                descriptor_df,
                morgan_fingerprint_df,
                maccs_key_df
            ]
        )  # Merge input_df to generated dfs for mordred descriptors,
        # Morgan fingerprints, and MACCS keys

        output_df = merged_df.drop(columns=['molecules'])  # Drop the
        # 'molecules' col since it's not very useful for later modeling

    return output_df
