# %%
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, PolynomialFeatures

try:
    from IPython.display import display, Image
except Exception:
    def display(obj):
        print(obj)

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except Exception:
    pass

# %%
# ---------------- Project paths ----------------
def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "data").exists():
            return p
    return start.parent

PROJECT_ROOT = find_project_root(Path(__file__).resolve())
DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "outputs" / "module1" / "figures"
TABLE_DIR = PROJECT_ROOT / "outputs" / "module1" / "tables"

MERGED_PATH = DATA_DIR / "reviews_with_listings.csv"

#%%
def save_and_show(fig_path: Path, fig=None, dpi: int = 220) -> None:
    # Always embed saved figure in cell output to avoid backend/display differences.
    if fig is None:
        plt.tight_layout()
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        fig.tight_layout()
        fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

    try:
        display(Image(filename=str(fig_path)))
    except Exception:
        pass
# %%
def feature_selection(merged: pd.DataFrame) -> tuple[list[str], list[str]]:
    CORR_PRUNE_THRESHOLD = 0.995

    numeric_candidates = [
        'accommodates', 'bathrooms', 'bedrooms', 'beds',
        'minimum_nights', 'maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm',
        'latitude', 'longitude',
        'availability_365', 'number_of_reviews', 'number_of_reviews_ltm',
        'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
        'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
        'review_scores_value', 'calculated_host_listings_count',
        'calculated_host_listings_count_entire_homes', 'calculated_host_listings_count_private_rooms',
        'host_tenure_days', 'host_response_rate_num', 'host_acceptance_rate_num',
        'amenities_count', 'review_count', 'mean_comment_length', 'review_active_days',
        'host_has_profile_pic_bin', 'host_identity_verified_bin', 'instant_bookable_bin',
    ]

    categorical_candidates = ['property_type', 'room_type', 'neighbourhood_cleansed', 'host_response_time']
    # Filter candidates to those present in the merged dataset and with non-missing values
    numeric_features = [c for c in numeric_candidates if c in merged.columns and merged[c].notna().sum() > 0]
    # For categorical features, we will later apply one-hot encoding but only to those with sufficient representation
    categorical_features = [c for c in categorical_candidates if c in merged.columns and merged[c].notna().sum() > 0]

    # Correlation-based pruning for near-duplicate numeric features
    num_imputed = merged[numeric_features].copy().fillna(merged[numeric_features].median(numeric_only=True))
    abs_corr = num_imputed.corr().abs()
    upper = abs_corr.where(np.triu(np.ones(abs_corr.shape), k=1).astype(bool))
    correlation_removed = [col for col in upper.columns if any(upper[col] > CORR_PRUNE_THRESHOLD)]
    selected_numeric = [c for c in numeric_features if c not in correlation_removed]

    return selected_numeric, categorical_features

def build_model_dataset(merged: pd.DataFrame) -> pd.DataFrame:
    # Initial filtering to remove entries with missing target and extreme outliers
    model_df = merged[merged['price_clean'].notna()].copy()
    price_cap_99 = float(model_df['price_clean'].quantile(0.99))
    model_df = model_df[model_df['price_clean'] <= price_cap_99].copy()

    return model_df

    # %%
    # ---------- Build model dataset and split --------
def main():
    # --------Load Cleaned merged dataset--------
    merged = pd.read_csv(MERGED_PATH)
    model_df = build_model_dataset(merged)
    selected_numeric, categorical_features = feature_selection(merged)
    feature_cols = selected_numeric + categorical_features

    X = model_df[feature_cols]
    y = model_df['price_clean']

    # Fixed split for fair comparison across all models
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Inner validation split for lightweight tuning
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    metrics_store = {}
    pred_store = {}
    model_store = {}
    meta_store = {}

    # ----------OLS baseline ----------
    base_preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
            ]), selected_numeric),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=0.01)),
            ]), categorical_features),
        ]
    )

    ols_pipe = Pipeline([
        ('preprocess', base_preprocessor),
        ('model', LinearRegression()),
    ])
    ols_pipe.fit(X_train_full, np.log1p(y_train_full))
    ols_log_pred = ols_pipe.predict(X_test)
    ols_pred = np.clip(np.expm1(np.clip(ols_log_pred, -5.0, 9.0)), 0, None)

    ols_metrics = {
        'r2': float(r2_score(y_test, ols_pred)),
        'mae': float(mean_absolute_error(y_test, ols_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, ols_pred))),
    }

    metrics_store['OLS'] = ols_metrics
    pred_store['OLS'] = ols_pred
    model_store['OLS'] = ols_pipe
    meta_store['OLS'] = {'family': 'linear', 'notes': 'baseline linear model'}

    display(pd.DataFrame([{'model': 'OLS', **ols_metrics}]))

    ols_residual = y_test.values - ols_pred
    plt.figure(figsize=(7, 4))
    plt.hist(ols_residual, bins=50)
    plt.xlabel('Residual (actual - predicted)')
    plt.ylabel('Count')
    plt.title('OLS Residual Distribution')
    save_and_show(FIG_DIR / 'ols_residual_hist.png')

    # ----------Ridge and ElasticNet----------
    ridge_candidates = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    ridge_trials = []
    best_ridge_alpha = None
    best_ridge_val_r2 = -np.inf

    for alpha in ridge_candidates:
        ridge_tmp = Pipeline([
            ('preprocess', base_preprocessor),
            ('model', Ridge(alpha=alpha, random_state=42)),
        ])
        ridge_tmp.fit(X_train, np.log1p(y_train))
        val_pred = np.clip(np.expm1(np.clip(ridge_tmp.predict(X_val), -5.0, 9.0)), 0, None)
        val_r2 = float(r2_score(y_val, val_pred))
        ridge_trials.append({'alpha': alpha, 'val_r2': val_r2})
        if val_r2 > best_ridge_val_r2:
            best_ridge_val_r2 = val_r2
            best_ridge_alpha = alpha

    ridge_pipe = Pipeline([
        ('preprocess', base_preprocessor),
        ('model', Ridge(alpha=best_ridge_alpha, random_state=42)),
    ])
    ridge_pipe.fit(X_train_full, np.log1p(y_train_full))
    ridge_pred = np.clip(np.expm1(np.clip(ridge_pipe.predict(X_test), -5.0, 9.0)), 0, None)

    ridge_metrics = {
        'r2': float(r2_score(y_test, ridge_pred)),
        'mae': float(mean_absolute_error(y_test, ridge_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, ridge_pred))),
    }

    metrics_store['Ridge'] = ridge_metrics
    pred_store['Ridge'] = ridge_pred
    model_store['Ridge'] = ridge_pipe
    meta_store['Ridge'] = {'family': 'linear', 'notes': f'alpha={best_ridge_alpha}'}

    elastic_grid = [
        (0.0003, 0.5),
        (0.0003, 0.7),
        (0.001, 0.5),
        (0.001, 0.7),
        (0.003, 0.7),
    ]
    enet_trials = []
    best_enet_cfg = None
    best_enet_val_r2 = -np.inf

    for alpha, l1 in elastic_grid:
        enet_tmp = Pipeline([
            ('preprocess', base_preprocessor),
            ('model', ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=12000, random_state=42)),
        ])
        enet_tmp.fit(X_train, np.log1p(y_train))
        val_pred = np.clip(np.expm1(np.clip(enet_tmp.predict(X_val), -5.0, 9.0)), 0, None)
        val_r2 = float(r2_score(y_val, val_pred))
        enet_trials.append({'alpha': alpha, 'l1_ratio': l1, 'val_r2': val_r2, 'config': f'a={alpha}, l1={l1}'})
        if val_r2 > best_enet_val_r2:
            best_enet_val_r2 = val_r2
            best_enet_cfg = (alpha, l1)

    best_enet_alpha, best_enet_l1 = best_enet_cfg
    enet_pipe = Pipeline([
        ('preprocess', base_preprocessor),
        ('model', ElasticNet(alpha=best_enet_alpha, l1_ratio=best_enet_l1, max_iter=12000, random_state=42)),
    ])
    enet_pipe.fit(X_train_full, np.log1p(y_train_full))
    enet_pred = np.clip(np.expm1(np.clip(enet_pipe.predict(X_test), -5.0, 9.0)), 0, None)

    enet_metrics = {
        'r2': float(r2_score(y_test, enet_pred)),
        'mae': float(mean_absolute_error(y_test, enet_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, enet_pred))),
    }

    metrics_store['ElasticNet'] = enet_metrics
    pred_store['ElasticNet'] = enet_pred
    model_store['ElasticNet'] = enet_pipe
    meta_store['ElasticNet'] = {'family': 'linear', 'notes': f'alpha={best_enet_alpha}, l1_ratio={best_enet_l1}'}

    display(pd.DataFrame([
        {'model': 'Ridge', **ridge_metrics},
        {'model': 'ElasticNet', **enet_metrics},
    ]))

    ridge_df = pd.DataFrame(ridge_trials)
    enet_df = pd.DataFrame(enet_trials)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(ridge_df['alpha'], ridge_df['val_r2'], marker='o')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('alpha (log scale)')
    axes[0].set_ylabel('Validation R²')
    axes[0].set_title('Ridge Validation Curve')

    axes[1].bar(enet_df['config'], enet_df['val_r2'])
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_ylabel('Validation R²')
    axes[1].set_title('ElasticNet Validation Scores')

    save_and_show(FIG_DIR / 'Ridge and ElasticNet Validation.png', fig=fig)

    # ----------PolyRidge----------
    poly_candidates = [
        'accommodates', 'bathrooms', 'bedrooms', 'beds',
        'latitude', 'longitude', 'review_scores_rating',
        'availability_365', 'amenities_count'
    ]
    poly_numeric = [c for c in poly_candidates if c in selected_numeric]
    rest_numeric = [c for c in selected_numeric if c not in poly_numeric]

    poly_preprocessor = ColumnTransformer(
        transformers=[
            ('poly_num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ]), poly_numeric),
            ('rest_num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
            ]), rest_numeric),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=0.01)),
            ]), categorical_features),
        ]
    )

    poly_ridge_candidates = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    poly_trials = []
    best_poly_alpha = None
    best_poly_val_r2 = -np.inf

    for alpha in poly_ridge_candidates:
        poly_tmp = Pipeline([
            ('preprocess', poly_preprocessor),
            ('model', Ridge(alpha=alpha, random_state=42)),
        ])
        poly_tmp.fit(X_train, np.log1p(y_train))
        val_pred = np.clip(np.expm1(np.clip(poly_tmp.predict(X_val), -5.0, 9.0)), 0, None)
        val_r2 = float(r2_score(y_val, val_pred))
        poly_trials.append({'alpha': alpha, 'val_r2': val_r2})
        if val_r2 > best_poly_val_r2:
            best_poly_val_r2 = val_r2
            best_poly_alpha = alpha

    poly_ridge_pipe = Pipeline([
        ('preprocess', poly_preprocessor),
        ('model', Ridge(alpha=best_poly_alpha, random_state=42)),
    ])
    poly_ridge_pipe.fit(X_train_full, np.log1p(y_train_full))
    poly_ridge_pred = np.clip(np.expm1(np.clip(poly_ridge_pipe.predict(X_test), -5.0, 9.0)), 0, None)

    poly_metrics = {
        'r2': float(r2_score(y_test, poly_ridge_pred)),
        'mae': float(mean_absolute_error(y_test, poly_ridge_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, poly_ridge_pred))),
    }

    metrics_store['PolyRidge'] = poly_metrics
    pred_store['PolyRidge'] = poly_ridge_pred
    model_store['PolyRidge'] = poly_ridge_pipe
    meta_store['PolyRidge'] = {'family': 'linear', 'notes': f'degree=2, alpha={best_poly_alpha}'}

    display(pd.DataFrame([{'model': 'PolyRidge', **poly_metrics}]))

    poly_df = pd.DataFrame(poly_trials)
    plt.figure(figsize=(7, 4.2))
    plt.plot(poly_df['alpha'], poly_df['val_r2'], marker='o')
    plt.xscale('log')
    plt.xlabel('alpha (log scale)')
    plt.ylabel('Validation R²')
    plt.title('PolyRidge Validation Curve')
    save_and_show(FIG_DIR / 'polyridge validation.png')

    # ----------Tree Ensembles----------
    tree_ohe_preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
            ]), selected_numeric),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=0.01)),
            ]), categorical_features),
        ]
    )

    rf_pipe = Pipeline([
        ('preprocess', tree_ohe_preprocessor),
        ('model', RandomForestRegressor(
            n_estimators=700,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        )),
    ])
    rf_pipe.fit(X_train_full, y_train_full)
    rf_pred = np.clip(rf_pipe.predict(X_test), 0, None)
    rf_metrics = {
        'r2': float(r2_score(y_test, rf_pred)),
        'mae': float(mean_absolute_error(y_test, rf_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, rf_pred))),
    }
    metrics_store['RandomForest'] = rf_metrics
    pred_store['RandomForest'] = rf_pred
    model_store['RandomForest'] = rf_pipe
    meta_store['RandomForest'] = {'family': 'tree', 'notes': 'n_estimators=700, min_samples_leaf=2'}

    et_pipe = Pipeline([
        ('preprocess', tree_ohe_preprocessor),
        ('model', ExtraTreesRegressor(
            n_estimators=1000,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=42,
        )),
    ])
    et_pipe.fit(X_train_full, y_train_full)
    et_pred = np.clip(et_pipe.predict(X_test), 0, None)
    et_metrics = {
        'r2': float(r2_score(y_test, et_pred)),
        'mae': float(mean_absolute_error(y_test, et_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, et_pred))),
    }
    metrics_store['ExtraTrees'] = et_metrics
    pred_store['ExtraTrees'] = et_pred
    model_store['ExtraTrees'] = et_pipe
    meta_store['ExtraTrees'] = {'family': 'tree', 'notes': 'n_estimators=1000, min_samples_leaf=1'}

    tree_ordinal_preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
            ]), selected_numeric),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
            ]), categorical_features),
        ]
    )

    hgb_pipe = Pipeline([
        ('preprocess', tree_ordinal_preprocessor),
        ('model', HistGradientBoostingRegressor(
            max_depth=10,
            learning_rate=0.05,
            max_iter=1200,
            min_samples_leaf=20,
            random_state=42,
        )),
    ])
    hgb_pipe.fit(X_train_full, y_train_full)
    hgb_pred = np.clip(hgb_pipe.predict(X_test), 0, None)
    hgb_metrics = {
        'r2': float(r2_score(y_test, hgb_pred)),
        'mae': float(mean_absolute_error(y_test, hgb_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, hgb_pred))),
    }
    metrics_store['HistGradientBoosting'] = hgb_metrics
    pred_store['HistGradientBoosting'] = hgb_pred
    model_store['HistGradientBoosting'] = hgb_pipe
    meta_store['HistGradientBoosting'] = {
        'family': 'tree',
        'notes': 'max_depth=10, learning_rate=0.05, max_iter=1200',
    }

    tree_summary = pd.DataFrame([
        {'model': 'RandomForest', **rf_metrics},
        {'model': 'ExtraTrees', **et_metrics},
        {'model': 'HistGradientBoosting', **hgb_metrics},
    ]).sort_values('r2', ascending=False)

    display(tree_summary)

    plt.figure(figsize=(7.5, 4.2))
    plt.bar(tree_summary['model'], tree_summary['r2'])
    plt.ylabel('R²')
    plt.title('Tree Models: Test R² Comparison')
    save_and_show(FIG_DIR / 'tree model comparison.png')

    # --------Final Model Comparison and Selection--------
    model_rows = []
    for model_name, m in metrics_store.items():
        row = {
            'model': model_name,
            'r2': m['r2'],
            'mae': m['mae'],
            'rmse': m['rmse'],
            'family': meta_store[model_name]['family'],
            'notes': meta_store[model_name]['notes'],
        }
        model_rows.append(row)

    model_results = pd.DataFrame(model_rows).sort_values('r2', ascending=False).reset_index(drop=True)
    display(model_results)

    plt.figure(figsize=(8.5, 4.5))
    plt.bar(model_results['model'], model_results['r2'])
    plt.ylim(max(0.0, model_results['r2'].min() - 0.05), min(1.0, model_results['r2'].max() + 0.05))
    plt.ylabel('R²')
    plt.title('Model Comparison by R²')
    plt.xticks(rotation=20, ha='right')
    save_and_show(FIG_DIR / 'model comparison r2.png')

    best_model_name = model_results.iloc[0]['model']
    best_model_family = model_results.iloc[0]['family']
    best_pipe = model_store[best_model_name]
    best_pred = pred_store[best_model_name]

    print('Selected best model by R²:', best_model_name)
    print('Best model family:', best_model_family)
    print('Target metric:', 'r2')
    print('One-hot rare-category threshold:', 0.01)

    # ---------Best-Model Diagnostics and Coefficients---------
    from sklearn.inspection import permutation_importance

    pred_df = pd.DataFrame({'actual_price': y_test.values, 'predicted_price': best_pred})
    pred_df['residual'] = pred_df['actual_price'] - pred_df['predicted_price']

    plt.figure(figsize=(6, 6))
    plt.scatter(pred_df['actual_price'], pred_df['predicted_price'], alpha=0.35)
    lims = [0, float(max(pred_df['actual_price'].max(), pred_df['predicted_price'].max()))]
    plt.plot(lims, lims, 'r--')
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel('Actual price')
    plt.ylabel('Predicted price')
    plt.title(f'{best_model_name}: Actual vs Predicted')
    save_and_show(FIG_DIR / 'fig_regression_actual_vs_pred.png')

    plt.figure(figsize=(7, 4))
    plt.hist(pred_df['residual'], bins=50)
    plt.xlabel('Residual (actual - predicted)')
    plt.ylabel('Count')
    plt.title(f'{best_model_name}: Residual Distribution')
    save_and_show(FIG_DIR / 'fig_regression_residual_hist.png')

    plt.figure(figsize=(7, 4))
    plt.scatter(pred_df['predicted_price'], pred_df['residual'], alpha=0.35)
    plt.axhline(0.0, color='red', linestyle='--')
    plt.xlabel('Predicted price')
    plt.ylabel('Residual')
    plt.title(f'{best_model_name}: Residual vs Predicted')
    save_and_show(FIG_DIR / 'fig_regression_residual_vs_pred.png')

    if best_model_family == 'linear':
        feature_names = best_pipe.named_steps['preprocess'].get_feature_names_out()
        coefs = best_pipe.named_steps['model'].coef_
        coef_df = pd.DataFrame({'feature': feature_names, 'coefficient': coefs, 'abs_coefficient': np.abs(coefs)}).sort_values('abs_coefficient', ascending=False)

        top_plot = coef_df.head(15).sort_values('abs_coefficient', ascending=True)
        plt.figure(figsize=(9, 5))
        plt.barh(top_plot['feature'], top_plot['abs_coefficient'])
        plt.xlabel('Absolute coefficient')
        plt.ylabel('Feature')
        plt.title(f'{best_model_name}: Top 15 Absolute Coefficients')
        save_and_show(FIG_DIR / 'fig_regression_top_coef.png')

        display(coef_df.head(15))
    else:
        perm = permutation_importance(
            best_pipe,
            X_test,
            y_test,
            n_repeats=8,
            random_state=42,
            n_jobs=-1,
            scoring='r2',
        )
        coef_df = pd.DataFrame({
            'feature': X_test.columns,
            'importance_mean': perm.importances_mean,
            'importance_std': perm.importances_std,
            'abs_importance': np.abs(perm.importances_mean),
        }).sort_values('abs_importance', ascending=False)

        top_plot = coef_df.head(15).sort_values('abs_importance', ascending=True)
        plt.figure(figsize=(9, 5))
        plt.barh(top_plot['feature'], top_plot['abs_importance'])
        plt.xlabel('Permutation importance (|ΔR²|)')
        plt.ylabel('Feature')
        plt.title(f'{best_model_name}: Top 15 Feature Importances')
        save_and_show(FIG_DIR / 'fig_regression_top_importance.png')

        display(coef_df.head(15))

    plt.figure(figsize=(6, 6))
    plt.scatter(pred_df['actual_price'], pred_df['predicted_price'], alpha=0.35)

    zoom_max = 1000
    plt.plot([0, zoom_max], [0, zoom_max], 'r--')

    plt.xlim(0, zoom_max)
    plt.ylim(0, zoom_max)

    plt.xlabel('Actual price')
    plt.ylabel('Predicted price')
    plt.title(f'{best_model_name}: Actual vs Predicted (<$1000)')
    save_and_show(FIG_DIR / 'fig_regression_actual_vs_pred_zoom1000.png')
# %%
if __name__ == "__main__":
    main()