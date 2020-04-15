import pandas as pd
import pickle as pkl

from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import PandasTools
from rdkit import Chem
import numpy as np
from tqdm.notebook import tqdm


def smile2unfold(smile):
   
    '''

    Given a SMILE Code,
    convert to unfolded morgan fingerprint

    ''' 

    bi = {}
    Chem.AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smile), radius=1, bitInfo=bi)

    # Iterate over dictionary to store as pd.Series
    mol = pd.DataFrame()
    for key, value in bi.items():
        for i in range(len(value)):
            sub = pd.Series()
            sub['val'] = str(key)

            sub['order'] = value[i][0]
            sub['radius'] = value[i][1]

            mol = mol.append(sub, ignore_index=True)

    # Correctly order fingerprint
    mol = mol.sort_values(by=['order', 'radius']).reset_index(drop=True)

    # Convert ordered fingerprint into single string
    mol_string = " ".join(mol.val.tolist())
        
    return mol_string.strip()



def trim_padding(matrices, target_length = None):
    '''
    Trim the X of -1's. Might loose actual sequences.
    Input X output, trimmed X
    '''

    padding_fractions = []
    padding_fraction = 0.9

    if not target_length:
    
        while padding_fraction > 0.5:

            padding_fraction = len(matrices[0][np.where(matrices[0] == -1)]) / (matrices[0].shape[0] * matrices[0].shape[1])
            padding_fractions.append(padding_fraction)
            trimming_factor = 0.1
            
            for i, m in enumerate(matrices):
                matrices[i] = m[:,int(m.shape[1]*trimming_factor) : m.shape[1] - int(m.shape[1]*trimming_factor)]
        
    if target_length:

        left_margin = int((matrices[0].shape[1]/2) - (target_length/2))
        right_margin = int((matrices[0].shape[1]/2) + (target_length/2))

        print(left_margin)
        print(right_margin)

        for i, m in enumerate(matrices):
            matrices[i] = m[:, left_margin : right_margin]
        

    return matrices



def gene2ordinal(mode = 'amino'):

    gene_sample = 'ATGGTCTCAGGAACGG'
    amino_sample = ''

    label_encoder = LabelEncoder()
    
    if mode == 'gene':
        label_encoder.fit(gene_sample)
    if mode == 'amino':
        label_encoder.fit(amino_sample)


def pad_inputs(inputs):
        
        maxlen = max(len(i) for i in inputs)

        if not isinstance(inputs[0], list):
            inputs = [i for i in inputs.strip()]

        
        padded = [int( (maxlen - len(i)) / 2 ) * [-1] + i + int( (maxlen - len(i)) / 2 ) * [-1] for i in inputs]
        padded = [i + (maxlen - len(i)) * [-1] for i in padded]


        return padded


       
def DDR_df_pad_fingerprint(df):


    df['drug_fingerprint_encoded'] = df['fingerprint']\
                                            .apply(lambda x: [int(i) for i in x.replace(' ', '')])

    df['drug_fingerprint_encoded_len'] = df['drug_fingerprint_encoded']\
                                            .apply(lambda x: len(x))

    max_drug_fingerprint_encoded_len = max(df['drug_fingerprint_encoded_len'])
    
    df['drug_fingerprint_encoded_mismatch'] = max_drug_fingerprint_encoded_len - df['drug_fingerprint_encoded_len'] 

    df['drug_fingerprint_encoded_padded'] = df[['drug_fingerprint_encoded','drug_fingerprint_encoded_mismatch']]\
                                                                 .apply(lambda x: ['-1' for i in range(int(x['drug_fingerprint_encoded_mismatch']/2))] +   
                                                                                list(x['drug_fingerprint_encoded']) +
                                                                                ['-1' for i in range (int(x['drug_fingerprint_encoded_mismatch']/2))]
                                                                                , axis = 1)

    df['drug_fingerprint_encoded_padded'] = df['drug_fingerprint_encoded_padded'].apply(lambda x: x + ['-1'] if len(x) < max_drug_fingerprint_encoded_len else x)

    return df


def DDR_df_pad_amino(df):


    with open('data/label_encoders/amino_encoder.pkl', 'rb') as f:

        label_encoder = pkl.load(f)

    amino_chars = set(list(df['amino'].iloc[0]))


    df['target_amino_encoded'] = df['amino'].apply(\
                                   lambda x: label_encoder.transform([i for i in list(x) if i in amino_chars]))

    df['target_amino_encoded_len'] = df['target_amino_encoded']\
                                    .apply(lambda x: len(x))


    max_target_amino_encoded_len = max(df['target_amino_encoded_len'])


    df['target_amino_encoded_mismatch'] = max_target_amino_encoded_len - df['target_amino_encoded_len']


    df['target_amino_encoded_padded'] = df[['target_amino_encoded','target_amino_encoded_mismatch']]\
                                         .apply(lambda x: ['-1' for i in range(int(x['target_amino_encoded_mismatch']/2))] +   
                                                        list(x['target_amino_encoded']) +
                                                        ['-1' for i in range (int(x['target_amino_encoded_mismatch']/2))]
                                                        , axis = 1)
    df['target_amino_encoded_padded'] = df['target_amino_encoded_padded'].apply(lambda x: x + ['-1'] if len(x) < max_target_amino_encoded_len else x)

    return df

 
def DDR_df_pad_gene(df):


    with open('data/label_encoders/gene_encoder.pkl', 'rb') as f:

        label_encoder = pkl.load(f)

    gene_chars = set(list(df['gene'].iloc[0]))

    df['target_gene_encoded'] = df['gene'].apply(\
                                   lambda x: label_encoder.transform([i for i in list(x) if i in gene_chars]))

    df['target_gene_encoded_len'] = df['target_gene_encoded']\
                                    .apply(lambda x: len(x))


    max_target_gene_encoded_len = max(df['target_gene_encoded_len'])


    df['target_gene_encoded_mismatch'] = max_target_gene_encoded_len - df['target_gene_encoded_len']


    df['target_gene_encoded_padded'] = df[['target_gene_encoded','target_gene_encoded_mismatch']]\
                                         .apply(lambda x: ['-1' for i in range(int(x['target_gene_encoded_mismatch']/2))] +   
                                                        list(x['target_gene_encoded']) +
                                                        ['-1' for i in range (int(x['target_gene_encoded_mismatch']/2))]
                                                        , axis = 1)
    df['target_gene_encoded_padded'] = df['target_gene_encoded_padded'].apply(lambda x: x + ['-1'] if len(x) < max_target_gene_encoded_len else x)

   
    return df


def generate_pairs(X_0, X_1):

    X = np.ones((X_0.shape[0] * X_1.shape[0], X_0.shape[1] + X_1.shape[1]))
    #X = np.zeros((X_0.shape[0] * X_1.shape[0], target_length))

    for i in tqdm(range(X_0.shape[0])):

        for j in range(X_1.shape[0]):
            
            X[int((X_1.shape[0]*i)+j),:] = np.hstack((X_0[i,:], X_1[j,:]))

    return X

