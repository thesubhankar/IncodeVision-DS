"""
Task-01: Data Cleaning and Preprocessing Pipeline
Modular, production-ready pipeline for cleaning arbitrary datasets.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder


class DataCleaningPipeline:
    """
    A robust, automated and customizable Data Cleaning & Preprocessing Pipeline.
    Works seamlessly on any CSV or Excel dataset.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.audit_log: List[Dict[str, Any]] = []
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = None

    def log_step(self, step_name: str, details: str, stats: Optional[Dict[str, Any]] = None):
        """Record an operation in the audit trail."""
        self.audit_log.append({
            "step": step_name,
            "details": details,
            "stats": stats or {}
        })

    @staticmethod
    def get_dataset_health(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive diagnostic metrics for a dataset."""
        total_rows, total_cols = df.shape
        missing_counts = df.isnull().sum()
        total_missing = int(missing_counts.sum())
        total_cells = total_rows * total_cols if total_rows * total_cols > 0 else 1
        missing_percentage = round((total_missing / total_cells) * 100, 2)

        duplicate_rows = int(df.duplicated().sum())

        col_diagnostics = []
        for col in df.columns:
            null_count = int(missing_counts[col])
            null_pct = round((null_count / total_rows * 100), 2) if total_rows > 0 else 0
            dtype = str(df[col].dtype)
            n_unique = int(df[col].nunique(dropna=True))

            issues = []
            if null_count > 0:
                issues.append(f"{null_count} missing ({null_pct}%)")
            if dtype == 'object':
                str_sample = df[col].dropna().astype(str)
                if any(s != s.strip() for s in str_sample.iloc[:100]):
                    issues.append("Trailing/leading whitespace detected")

            col_diagnostics.append({
                "Column": col,
                "Type": dtype,
                "Null Count": null_count,
                "Null %": null_pct,
                "Unique Values": n_unique,
                "Detected Issues": ", ".join(issues) if issues else "Clean"
            })

        return {
            "total_rows": total_rows,
            "total_cols": total_cols,
            "total_missing_values": total_missing,
            "missing_percentage": missing_percentage,
            "duplicate_rows": duplicate_rows,
            "columns": col_diagnostics
        }

    def remove_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        keep: str = 'first'
    ) -> pd.DataFrame:
        """Remove full or subset-based duplicate records."""
        initial_len = len(df)
        df_clean = df.drop_duplicates(subset=subset, keep=keep).copy()
        dropped = initial_len - len(df_clean)

        self.log_step(
            "Duplicate Removal",
            f"Removed {dropped} duplicate rows (subset: {subset or 'all columns'})",
            {"initial_rows": initial_len, "final_rows": len(df_clean), "duplicates_removed": dropped}
        )
        return df_clean

    def fix_data_formats(
        self,
        df: pd.DataFrame,
        strip_whitespace: bool = True,
        auto_boolean_conversion: bool = True,
        auto_numeric_conversion: bool = True
    ) -> pd.DataFrame:
        """Trim whitespace, parse boolean literals, and fix common string-number representations."""
        df_clean = df.copy()
        modifications = 0

        for col in df_clean.columns:
            if df_clean[col].dtype == 'object' or pd.api.types.is_string_dtype(df_clean[col]):
                if strip_whitespace:
                    df_clean[col] = df_clean[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

                if auto_boolean_conversion:
                    unique_vals = set(df_clean[col].dropna().astype(str).str.strip().str.lower().unique())
                    boolean_map = {
                        'true': True, 'false': False,
                        'yes': True, 'no': False,
                        'y': True, 'n': False,
                        '1': True, '0': False,
                        't': True, 'f': False
                    }
                    if unique_vals.issubset(boolean_map.keys()) and len(unique_vals) > 0:
                        df_clean[col] = df_clean[col].astype(str).str.strip().str.lower().map(boolean_map)
                        modifications += 1

                if auto_numeric_conversion and (df_clean[col].dtype == 'object'):
                    try:
                        sample = df_clean[col].dropna().astype(str).str.strip().str.replace(r'[\$,%]', '', regex=True)
                        converted = pd.to_numeric(sample, errors='coerce')
                        if converted.notnull().mean() > 0.85 and converted.notnull().sum() > 0:
                            cleaned_col = df_clean[col].astype(str).str.strip().str.replace(r'[\$,%]', '', regex=True)
                            df_clean[col] = pd.to_numeric(cleaned_col, errors='coerce')
                            modifications += 1
                    except Exception:
                        pass

        self.log_step(
            "Format & Type Normalization",
            f"Normalized whitespaces, types, and string patterns across {len(df_clean.columns)} columns.",
            {"converted_columns": modifications}
        )
        return df_clean

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        numeric_strategy: str = "median",
        categorical_strategy: str = "mode",
        custom_imputations: Optional[Dict[str, Any]] = None,
        drop_col_threshold: float = 0.8
    ) -> pd.DataFrame:
        df_clean = df.copy()
        custom_imputations = custom_imputations or {}
        total_initial_nulls = int(df_clean.isnull().sum().sum())

        dropped_cols = []
        for col in df_clean.columns:
            if df_clean[col].isnull().mean() > drop_col_threshold:
                dropped_cols.append(col)
        if dropped_cols:
            df_clean = df_clean.drop(columns=dropped_cols)

        imputed_stats = {}
        for col in df_clean.columns:
            null_count = int(df_clean[col].isnull().sum())
            if null_count == 0:
                continue

            col_strat = custom_imputations.get(col, None)

            if pd.api.types.is_numeric_dtype(df_clean[col]):
                strat = col_strat or numeric_strategy
                if strat == "median":
                    fill_val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(fill_val)
                    imputed_stats[col] = f"Median ({fill_val:.2f})"
                elif strat == "mean":
                    fill_val = df_clean[col].mean()
                    df_clean[col] = df_clean[col].fillna(fill_val)
                    imputed_stats[col] = f"Mean ({fill_val:.2f})"
                elif strat == "mode":
                    mode_val = df_clean[col].mode()
                    fill_val = mode_val[0] if not mode_val.empty else 0
                    df_clean[col] = df_clean[col].fillna(fill_val)
                    imputed_stats[col] = f"Mode ({fill_val})"
                elif strat == "zero":
                    df_clean[col] = df_clean[col].fillna(0)
                    imputed_stats[col] = "Constant (0)"
                elif strat == "drop":
                    df_clean = df_clean.dropna(subset=[col])
                    imputed_stats[col] = f"Dropped {null_count} rows"
            else:
                strat = col_strat or categorical_strategy
                if strat == "mode":
                    mode_val = df_clean[col].mode()
                    fill_val = mode_val[0] if not mode_val.empty else "Unknown"
                    df_clean[col] = df_clean[col].fillna(fill_val)
                    imputed_stats[col] = f"Mode ('{fill_val}')"
                elif strat == "unknown":
                    df_clean[col] = df_clean[col].fillna("Unknown")
                    imputed_stats[col] = "Filled with 'Unknown'"
                elif strat == "drop":
                    df_clean = df_clean.dropna(subset=[col])
                    imputed_stats[col] = f"Dropped {null_count} rows"

        remaining_nulls = int(df_clean.isnull().sum().sum())
        self.log_step(
            "Missing Value Imputation",
            f"Handled {total_initial_nulls - remaining_nulls} missing entries across {len(imputed_stats)} columns.",
            {"imputed_columns": imputed_stats, "dropped_columns": dropped_cols, "remaining_nulls": remaining_nulls}
        )
        return df_clean

    def handle_outliers(
        self,
        df: pd.DataFrame,
        method: str = "iqr",
        factor: float = 1.5,
        action: str = "clip",
        exclude_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        if action == "none":
            return df

        df_clean = df.copy()
        exclude_columns = exclude_columns or []
        outlier_summary = {}
        rows_to_drop = set()

        numeric_cols = [
            c for c in df_clean.select_dtypes(include=[np.number]).columns
            if c not in exclude_columns and df_clean[c].nunique() > 10
        ]

        for col in numeric_cols:
            series = df_clean[col].dropna()
            if len(series) == 0:
                continue

            if method == "iqr":
                q25, q75 = series.quantile(0.25), series.quantile(0.75)
                iqr = q75 - q25
                lower_bound = q25 - factor * iqr
                upper_bound = q75 + factor * iqr
            else:
                mean, std = series.mean(), series.std()
                if std == 0:
                    continue
                lower_bound = mean - factor * std
                upper_bound = mean + factor * std

            outliers_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            num_outliers = int(outliers_mask.sum())

            if num_outliers > 0:
                outlier_summary[col] = {
                    "count": num_outliers,
                    "lower_bound": round(float(lower_bound), 2),
                    "upper_bound": round(float(upper_bound), 2)
                }

                if action == "clip":
                    df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
                elif action == "drop":
                    rows_to_drop.update(df_clean[outliers_mask].index.tolist())

        if action == "drop" and rows_to_drop:
            df_clean = df_clean.drop(index=list(rows_to_drop)).reset_index(drop=True)
            self.log_step(
                "Outlier Handling",
                f"Dropped {len(rows_to_drop)} rows with extreme outliers ({method.upper()} method).",
                {"dropped_rows": len(rows_to_drop), "cols_affected": outlier_summary}
            )
        elif action == "clip":
            self.log_step(
                "Outlier Handling",
                f"Clipped/winsorized extreme values across {len(outlier_summary)} numeric columns ({method.upper()} bounds).",
                {"clipped_columns": outlier_summary}
            )

        return df_clean

    def preprocess_features(
        self,
        df: pd.DataFrame,
        encode_categorical: bool = False,
        encoding_method: str = "label",
        scale_numeric: bool = False,
        scaling_method: str = "standard",
        exclude_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        df_clean = df.copy()
        exclude_columns = exclude_columns or []

        if encode_categorical:
            cat_cols = [
                c for c in df_clean.select_dtypes(include=['object', 'category', 'bool']).columns
                if c not in exclude_columns
            ]
            if encoding_method == "label":
                for col in cat_cols:
                    le = LabelEncoder()
                    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                    self.label_encoders[col] = le
                self.log_step("Feature Encoding", f"Label-encoded {len(cat_cols)} categorical columns.", {"encoded_columns": cat_cols})
            elif encoding_method == "onehot":
                df_clean = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)
                self.log_step("Feature Encoding", f"One-Hot encoded {len(cat_cols)} categorical columns.", {"encoded_columns": cat_cols})

        if scale_numeric:
            num_cols = [
                c for c in df_clean.select_dtypes(include=[np.number]).columns
                if c not in exclude_columns and c not in self.label_encoders
            ]
            if num_cols:
                if scaling_method == "standard":
                    scaler = StandardScaler()
                elif scaling_method == "minmax":
                    scaler = MinMaxScaler()
                else:
                    scaler = RobustScaler()

                df_clean[num_cols] = scaler.fit_transform(df_clean[num_cols])
                self.scaler = scaler
                self.log_step("Feature Scaling", f"Scaled {len(num_cols)} numeric columns using {scaling_method.title()}Scaler.", {"scaled_columns": num_cols})

        return df_clean

    def run_pipeline(
        self,
        df: pd.DataFrame,
        remove_dups: bool = True,
        dup_subset: Optional[List[str]] = None,
        fix_formats: bool = True,
        impute_nulls: bool = True,
        numeric_strategy: str = "median",
        categorical_strategy: str = "mode",
        handle_outliers: bool = True,
        outlier_method: str = "iqr",
        outlier_action: str = "clip",
        encode_cats: bool = False,
        encoding_method: str = "label",
        scale_nums: bool = False,
        scaling_method: str = "standard",
        id_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
        self.audit_log.clear()
        initial_health = self.get_dataset_health(df)
        data = df.copy()

        if remove_dups:
            data = self.remove_duplicates(data, subset=dup_subset)

        if fix_formats:
            data = self.fix_data_formats(data)

        if impute_nulls:
            data = self.handle_missing_values(
                data,
                numeric_strategy=numeric_strategy,
                categorical_strategy=categorical_strategy
            )

        if handle_outliers:
            data = self.handle_outliers(
                data,
                method=outlier_method,
                action=outlier_action,
                exclude_columns=id_columns
            )

        if encode_cats or scale_nums:
            data = self.preprocess_features(
                data,
                encode_categorical=encode_cats,
                encoding_method=encoding_method,
                scale_numeric=scale_nums,
                scaling_method=scaling_method,
                exclude_columns=id_columns
            )

        final_health = self.get_dataset_health(data)

        return data, self.audit_log, {
            "initial": initial_health,
            "final": final_health
        }
