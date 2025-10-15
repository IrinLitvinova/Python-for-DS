import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import Dict, Tuple, List, Any


def drop_unnecessary_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Drop columns that are not useful for model training."""
    return df.drop(columns=columns, axis=1)


def add_stratification_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add stratification columns for balanced train-test splitting."""
    df = df.copy()
    df.insert(df.shape[1] - 1, 'Balance_codes', df['Balance'].map(lambda x: 0 if x == 0 else 1))
    stratum = df['Balance_codes'].astype(str) + '_' + df['Exited'].astype(str)
    df.insert(df.shape[1] - 1, 'Stratum', stratum)
    return df


def split_data(
    df: pd.DataFrame, stratify_col: str, test_size: float = 0.25, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset into training and validation sets using stratified sampling."""
    return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[stratify_col])


def separate_inputs_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], str]:
    """Separate input features and target column."""
    input_cols = list(df.columns)[1:-2]
    target_col = list(df.columns)[-1]
    inputs = df[input_cols].copy()
    targets = df[target_col].copy()
    return inputs, targets, input_cols, target_col


def encode_gender(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert gender categorical values to numeric."""
    gender_map = {'Female': 0, 'Male': 1}
    for df in [train_df, val_df]:
        df['Gender'] = df['Gender'].map(gender_map).astype('int64')
    return train_df, val_df


def impute_numeric(train_df: pd.DataFrame, val_df: pd.DataFrame, numeric_cols: List[str]) -> SimpleImputer:
    """Impute missing numerical values with the mean."""
    imputer = SimpleImputer(strategy='mean').fit(train_df[numeric_cols])
    train_df[numeric_cols] = imputer.transform(train_df[numeric_cols])
    val_df[numeric_cols] = imputer.transform(val_df[numeric_cols])
    return imputer


def encode_categorical(train_df: pd.DataFrame, val_df: pd.DataFrame, categorical_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:
    """Apply one-hot encoding to categorical columns."""
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train_df[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    train_df[encoded_cols] = encoder.transform(train_df[categorical_cols])
    val_df[encoded_cols] = encoder.transform(val_df[categorical_cols])
    return train_df, val_df, encoder, encoded_cols


def scale_numeric(train_df: pd.DataFrame, val_df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """Scale numeric columns using MinMaxScaler."""
    scaler = MinMaxScaler().fit(train_df[numeric_cols])
    train_df[numeric_cols] = scaler.transform(train_df[numeric_cols])
    val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])
    return train_df, val_df, scaler


def preprocess_customer_data(raw_df: pd.DataFrame, scale_numeric_data: bool = False) -> Dict[str, Any]:
    """
    High-level preprocessing pipeline for customer churn data.

    Args:
        raw_df (pd.DataFrame): Raw input data.
        scale_numeric_data (bool): Whether to apply MinMax scaling to numeric features.

    Returns:
        Dict[str, Any]: Processed data and fitted transformers.
    """
    df = drop_unnecessary_columns(raw_df, ['CustomerId', 'Surname'])
    df = add_stratification_columns(df)

    train_df, val_df = split_data(df, stratify_col='Stratum')

    train_inputs, train_targets, input_cols, _ = separate_inputs_targets(train_df)
    val_inputs, val_targets, _, _ = separate_inputs_targets(val_df)

    train_inputs, val_inputs = encode_gender(train_inputs, val_inputs)

    numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = train_inputs.select_dtypes(include='object').columns.tolist()

    imputer = impute_numeric(train_inputs, val_inputs, numeric_cols)
    train_inputs, val_inputs, encoder, encoded_cols = encode_categorical(train_inputs, val_inputs, categorical_cols)

    """ Select the columns to be used for training/prediction"""
    train_inputs = train_inputs[numeric_cols + encoded_cols]
    val_inputs = val_inputs[numeric_cols + encoded_cols]

    if scale_numeric_data:
        train_inputs, val_inputs, scaler = scale_numeric(train_inputs, val_inputs, numeric_cols)
    else:
        scaler = None

    return {
        'X_train': train_inputs,
        'train_targets': train_targets,
        'X_val': val_inputs,
        'val_targets': val_targets,
        'input_cols': input_cols,
        'scaler': scaler,
        'encoder': encoder,
        'imputer': imputer,
    }

# import pandas as pd
# import numpy as np
# from sklearn.impute import SimpleImputer
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# def preprocess_data (raw_df, scaler_numeric=False):  # scaler_numeric - використовувати чи ні масштабування для числових даних
#     raw_df = raw_df.drop(['CustomerId', 'Surname'], axis=1)


#     # Create training, validation and test sets
#     raw_df.insert(raw_df.shape[1]-1, 'Balance_codes', raw_df.Balance.map(lambda x: 0 if x==0 else 1))
#     stratum = raw_df['Balance_codes'].astype(str) + '_' + raw_df['Exited'].astype(str)
#     raw_df.insert(raw_df.shape[1]-1, 'Stratum', stratum)

#     train_df, val_df = train_test_split(raw_df, test_size=0.25, random_state=42, stratify=raw_df['Stratum'])


#     # Create inputs and targets
#     input_cols = list(train_df.columns)[1:-2]
#     target_col = list(train_df.columns)[-1]
#     train_inputs, train_targets = train_df[input_cols].copy(), train_df[target_col].copy()
#     val_inputs, val_targets = val_df[input_cols].copy(), val_df[target_col].copy()


#     # Categorical features from binary values ​​to binary numerics
#     Gender_codes = {'Female': 0, 'Male': 1}
#     train_inputs['Gender'] = train_inputs.Gender.map(Gender_codes).astype('int64')
#     val_inputs['Gender'] = val_inputs.Gender.map(Gender_codes).astype('int64')


#     # Identify numerical and categorical columns
#     numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
#     categorical_cols = train_inputs.select_dtypes('object').columns.tolist()


#     # Impute missing numerical values
#     imputer = SimpleImputer(strategy='mean').fit(train_inputs[numeric_cols])
#     train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
#     val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])


#     # One-hot encode categorical features
#     encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train_inputs[categorical_cols])
#     encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
#     train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
#     val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])


#     # Scale numeric features
#     scaler = MinMaxScaler().fit(train_inputs[numeric_cols])
#     train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
#     val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])

#     result = {
#         'X_train': train_inputs,
#         'train_targets': train_targets,
#         'X_val': train_inputs,
#         'val_targets': train_targets,
#         'input_cols' : input_cols,
#         'scaler': scaler,
#         'encoder': encoder
#     }

#     return result