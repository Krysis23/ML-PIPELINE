import os
import json
import uuid
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io,base64

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, SGDClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier, BaggingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import(
    accuracy_score, f1_score, roc_auc_score, classification_report, mean_squared_error, r2_score, mean_absolute_error
)

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


SESSIONS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'sessions')
os.makedirs(SESSIONS_DIR, exist_ok=True)


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.3, dpi=120)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64

def detect_problem_type(series):
    n_unique = series.nunique()
    if series.dtype == object or n_unique <= 2:
        return 'binary' if n_unique <= 2 else 'multiclass'
    if n_unique <= 20:
        return 'multiclass'
    return 'regression'

def infer_column_types(df,target_col):
    feature_cols = [c for c in df.columns if c != target_col]
    numeric_cols = df[feature_cols].select_dtypes(include=['number']).columns.tolist()
    cat_cols = df[feature_cols].select_dtypes(exclude=['number']).columns.tolist()

    for col in list(numeric_cols):
        if df[col].nunique() <= 10 and df[col].nunique() >= 2:
            cat_cols.append(col)
            numeric_cols.remove(col)
    return numeric_cols, cat_cols


def build_preprocessor(numeric_cols,cat_cols):
    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    transformers = []
    if numeric_cols:
        transformers.append(('num', numeric_pipe, numeric_cols))
    if cat_cols:
        transformers.append(('cat',cat_pipe,cat_cols))
    return ColumnTransformer(transformers=transformers, remainder='drop')


MODEL_REGISTRY = {
    'logistic_regression': {
        'label': 'Logistic Regression',
        'type': 'classification',
        'model': LogisticRegression(max_iter=1000, random_state=42),
        'params': {
            'model__C': [0.1, 1.0, 10.0],
            'model__penalty': ['l2']
        }
    },
    'ridge': {
        'label': 'Ridge Regression',
        'type': 'regression',
        'model': Ridge(),
        'params': {
            'model__alpha': [0.1, 1.0, 10.0]
        }
    },
    'lasso': {
        'label': 'Lasso Regression',
        'type': 'regression',
        'model': Lasso(),
        'params': {
            'model__alpha': [0.001, 0.01, 0.1]
        }
    },
    'sgd_classifier': {
        'label': 'SGD Classifier',
        'type': 'classification',
        'model': SGDClassifier(max_iter=1000, tol=1e-3, random_state=42),
        'params': {
            'model__alpha': [0.0001, 0.001],
            'model__loss': ['log_loss']
        }
    },
    'decision_tree': {
        'label': 'Decision Tree',
        'type': 'both',
        'model_clf': DecisionTreeClassifier(random_state=42),
        'model_reg': DecisionTreeRegressor(random_state=42),
        'params': {
            'model__max_depth': [None, 10, 20],
            'model__min_samples_leaf': [1, 3]
        }
    },
    'random_forest': {
        'label': 'Random Forest',
        'type': 'both',
        'model_clf': RandomForestClassifier(n_estimators=100, random_state=42),
        'model_reg': RandomForestRegressor(n_estimators=100, random_state=42),
        'params': {
            'model__n_estimators': [50, 100],
            'model__max_depth': [None, 10]
        }
    },
    'extra_trees': {
        'label': 'Extra Trees',
        'type': 'both',
        'model_clf': ExtraTreesClassifier(n_estimators=100, random_state=42),
        'model_reg': ExtraTreesRegressor(n_estimators=100, random_state=42),
        'params': {
            'model__n_estimators': [50, 100],
            'model__max_depth': [None, 10]
        }
    },
    'gradient_boosting': {
        'label': 'Gradient Boosting',
        'type': 'both',
        'model_clf': GradientBoostingClassifier(random_state=42),
        'model_reg': GradientBoostingRegressor(random_state=42),
        'params': {
            'model__n_estimators': [50, 100],
            'model__learning_rate': [0.05, 0.1]
        }
    },
    'adaboost': {
        'label': 'AdaBoost',
        'type': 'both',
        'model_clf': AdaBoostClassifier(random_state=42),
        'model_reg': AdaBoostRegressor(random_state=42),
        'params': {
            'model__n_estimators': [50, 100],
            'model__learning_rate': [0.05, 0.1]
        }
    },
    'knn': {
        'label': 'K-Nearest Neighbors',
        'type': 'both',
        'model_clf': KNeighborsClassifier(),
        'model_reg': KNeighborsRegressor(),
        'params': {
            'model__n_neighbors': [3, 5, 7],
            'model__weights': ['uniform', 'distance']
        }
    },
    'svm': {
        'label': 'Support Vector Machine',
        'type': 'both',
        'model_clf': SVC(probability=True, gamma='scale', random_state=42),
        'model_reg': SVR(gamma='scale'),
        'params': {
            'model__C': [0.1, 1.0, 10.0],
            'model__kernel': ['rbf']
        }
    },
    'naive_bayes': {
        'label': 'Gaussian Naive Bayes',
        'type': 'classification',
        'model': GaussianNB(),
        'params': {}
    },
    'gaussian_process': {
        'label': 'Gaussian Process',
        'type': 'both',
        'model_clf': GaussianProcessClassifier(),
        'model_reg': GaussianProcessRegressor(),
        'params': {
            'model__alpha': [0.01, 0.1]
        }
    },
    'lda': {
        'label': 'Linear Discriminant Analysis',
        'type': 'classification',
        'model': LinearDiscriminantAnalysis(),
        'params': {}
    },
    'qda': {
        'label': 'Quadratic Discriminant Analysis',
        'type': 'classification',
        'model': QuadraticDiscriminantAnalysis(),
        'params': {}
    },
    'bagging': {
        'label': 'Bagging',
        'type': 'both',
        'model_clf': BaggingClassifier(random_state=42),
        'model_reg': BaggingRegressor(random_state=42),
        'params': {
            'model__n_estimators': [10, 50],
            'model__max_samples': [0.7, 1.0]
        }
    }
}

if HAS_XGB:
    MODEL_REGISTRY['xgboost'] = {
        'label': 'XGBoost',
        'type': 'both',
        'model_clf': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0),
        'model_reg': XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        'params': {
            'model__n_estimators': [50, 100],
            'model__learning_rate': [0.05, 0.1]
        }
    }


def _select_model_for_problem(entry, problem_type):
    if problem_type == 'regression':
        if entry['type'] == 'classification':
            return None
        if entry['type'] == 'both':
            return entry.get('model_reg')
        return entry.get('model')
    if entry['type'] == 'regression':
        return None
    if entry['type'] == 'both':
        return entry.get('model_clf')
    return entry.get('model')


def get_filtered_models(problem_type):
    """
    Return only models compatible with the given problem type.
    
    Args:
        problem_type (str): 'regression', 'binary', or 'multiclass'
    
    Returns:
        dict: Filtered model registry with label, type, and key
    """
    filtered = {}
    for key, entry in MODEL_REGISTRY.items():
        # For regression problems, skip classification-only models
        if problem_type == 'regression':
            if entry['type'] == 'classification':
                continue
        # For classification problems, skip regression-only models
        else:  # binary or multiclass
            if entry['type'] == 'regression':
                continue
        
        filtered[key] = {
            'label': entry['label'],
            'type': entry['type']
        }
    
    return filtered


def _get_scoring(problem_type):
    if problem_type == 'binary':
        return 'roc_auc'
    if problem_type == 'multiclass':
        return 'f1_weighted'
    return 'r2'


def train_selected_models(preprocessor, X_train, y_train, X_test, y_test, selected_keys, problem_type, progress_callback=None):
    results = {}
    best_score = -np.inf
    best_pipeline = None
    best_model_name = None
    scoring = _get_scoring(problem_type)
    candidate_keys = [key for key in selected_keys if key in MODEL_REGISTRY]

    for idx, key in enumerate(candidate_keys):
        entry = MODEL_REGISTRY[key]
        estimator = _select_model_for_problem(entry, problem_type)
        if estimator is None:
            continue

        pipe = Pipeline([('preprocessor', preprocessor), ('model', estimator)])
        params = entry.get('params', {}) or {}
        best_params = None
        cv_score = None

        if params and len(X_train) <= 10000:
            try:
                search = GridSearchCV(pipe, param_grid=params, cv=5, scoring=scoring, n_jobs=-1)
                search.fit(X_train, y_train)
                pipe = search.best_estimator_
                best_params = search.best_params_
                cv_score = float(search.best_score_)
            except Exception:
                pipe.fit(X_train, y_train)
        elif params:
            pipe.fit(X_train, y_train)
        else:
            try:
                scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring=scoring, n_jobs=-1)
                cv_score = float(np.mean(scores))
            except Exception:
                cv_score = None
            pipe.fit(X_train, y_train)

        metrics = evaluate_model(pipe, X_test, y_test, problem_type)
        results[entry['label']] = {
            'metrics': metrics,
            'best_params': best_params,
            'cv_score': cv_score
        }

        score = metrics.get('ROC-AUC', metrics.get('F1', metrics.get('Accuracy', metrics.get('R2', 0))))
        if score > best_score:
            best_score = score
            best_pipeline = pipe
            best_model_name = entry['label']
        
        # Call progress callback if provided
        if progress_callback:
            progress = int((idx + 1) / len(candidate_keys) * 100)
            progress_callback(idx, entry['label'], progress)

    return results, best_pipeline, best_model_name


def evaluate_model(model, X_test, y_test, problem_type):
    y_pred = model.predict(X_test)
    if problem_type == 'regression':
        rmse = float(np.sqrt(mean_squared_error(y_test,y_pred)))
        return {
            'R2': round(float(r2_score(y_test,y_pred)), 4),
            'MAE': round(float(mean_absolute_error(y_test,y_pred)), 4),
            'RMSE': round(rmse, 4)
        }
    else:
        avg = 'binary' if problem_type == 'binary' else 'weighted'
        metrics = {
            'Accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
            'F1': round(float(f1_score(y_test, y_pred, average=avg, zero_division=0)), 4),
        }
        try:
            if problem_type == 'binary':
                proba = model.predict_proba(X_test)[:,1]
                metrics['ROC-AUC'] = round(float(roc_auc_score(y_test, proba)), 4)
        except Exception:
            pass
        return metrics
    

def generate_eda_plots(df, target_col):
    plots = {}

    if df.isnull().sum().sum() > 0:
        fig, ax = plt.subplots(figsize=(9,5))
        missing = df.isnull().mean().sort_values(ascending=False)
        missing = missing[missing > 0]
        sns.barplot(x=missing.values, y=missing.index, ax=ax, color='#4f86c6')
        ax.set_title('Missing Value Rate by Column', fontsize=12)
        ax.set_xlabel('Missing Rate')
        plots['missing'] = _fig_to_b64(fig)

    fig, ax = plt.subplots(figsize=(8,5))
    if df[target_col].dtype == object or df[target_col].nunique() <= 20:
        df[target_col].value_counts().plot(kind='bar', ax=ax, color='#4f86c6', edgecolor='white')
        ax.set_title('Target Distribution', fontsize=12)
        ax.set_xlabel(target_col)
        plt.xticks(rotation=45)
    else:
        ax.hist(df[target_col].dropna(), bins=30, color='#4f86c6',edgecolor='white')
        ax.set_title('Target Distribution', fontsize=12)
        ax.set_xlabel(target_col)
    plots['target'] = _fig_to_b64(fig)

    num_df = df.select_dtypes(include='number')
    if len(num_df.columns) >= 2:
        fig, ax = plt.subplots(figsize=(10,8))
        corr = num_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=len(corr) <= 12, fmt='.2f', cmap='coolwarm', ax=ax, linewidths=0.5, vmin=-1, vmax=1)
        ax.set_title('Correlation Matrix', fontsize=12)
        plots['correlation'] = _fig_to_b64(fig)

    return plots

def training_sessions(df, target_col, session_id=None, selected_keys=None):
    if session_id is None:
        session_id = str(uuid.uuid4())[:8]
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    df = df.dropna(subset=[target_col])
    if len(df) < 20:
        raise ValueError("Dataset too small - need at least 20 rows after dropping nulls.")
    
    problem_type = detect_problem_type(df[target_col])
    numeric_cols, cat_cols = infer_column_types(df, target_col)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    le = None
    if problem_type != 'regression':
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if problem_type != 'regression' else None
        )
    
    preprocessor = build_preprocessor(numeric_cols, cat_cols)
    default_model_keys = ['logistic_regression', 'random_forest', 'xgboost'] if problem_type != 'regression' else ['ridge', 'random_forest', 'xgboost']
    if not HAS_XGB and 'xgboost' in default_model_keys:
        default_model_keys.remove('xgboost')
    if selected_keys is None:
        selected_keys = default_model_keys

    selected_keys = [key for key in selected_keys if isinstance(key, str) and key.strip()]
    results, best_pipeline, best_model_name = train_selected_models(
        preprocessor, X_train, y_train, X_test, y_test, selected_keys, problem_type
    )

    if best_pipeline is None:
        raise ValueError('No valid models selected for this problem type.')

    eda_plots = generate_eda_plots(df, target_col)

    importance_plot = None

    try:
        raw_model = best_pipeline.named_steps['model']
        if hasattr(raw_model, 'feature_importances_'):
            pre = best_pipeline.named_steps['preprocessor']
            feat_names =[]
            for name_,trans, cols in pre.transformers_:
                if name_ == 'num':
                    feat_names.extend(cols)
                elif name_ == 'cat':
                    feat_names.extend(
                        trans.named_steps['encoder'].get_feature_names_out(cols).tolist()
                    )
            imps = raw_model.feature_importances_
            top_n = min(15, len(feat_names))
            idx = np.argsort(imps)[-top_n:]
            fig, ax = plt.subplots(figsize=(9,6))
            ax.barh([feat_names[i] for i in idx], imps[idx], color='#4f86c6')
            ax.set_title(f'Top {top_n} Features Importances ({best_model_name})', fontsize=11)
            importance_plot = _fig_to_b64(fig)
    except Exception:
        pass

    session_data = {
        'session_id': session_id,
        'target_col' : target_col,
        'problem_type': problem_type,
        'numeric_cols': numeric_cols,
        'cat_cols': cat_cols,
        'feature_cols': numeric_cols + cat_cols,
        'best_model_name': best_model_name,
        'results': results,
        'classes': le.classes_.tolist() if le else None,
        'row_count': len(df),
        'col_count': len(df.columns),
    }

    session_path = os.path.join(SESSIONS_DIR, session_id)
    os.makedirs(session_path, exist_ok=True)

    joblib.dump(best_pipeline, os.path.join(session_path, 'pipeline.pkl'))
    if le:
        joblib.dump(le, os.path.join(session_path,'label_encoder.pkl'))
    with open(os.path.join(session_path, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(session_data, f)
    
    return {
        'session_id': session_id,
        'problem_type': problem_type,
        'best_model': best_model_name,
        'results': results,
        'eda_plots': eda_plots,
        'importance_plot': importance_plot,
        'meta': session_data
    }

def load_session(session_id):
    session_path = os.path.join(SESSIONS_DIR, session_id)
    meta_path = os.path.join(session_path, 'meta.json')
    pipeline_path = os.path.join(session_path, 'pipeline.pkl')

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Session '{session_id}' not found.")
    
    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)

    pipeline = joblib.load(pipeline_path)

    le = None
    le_path = os.path.join(session_path, 'label_encoder.pkl')
    if os.path.exists(le_path):
        le = joblib.load(le_path)
    
    return pipeline, le, meta


def predict_single(session_id, input_dict):
    pipeline, le, meta = load_session(session_id)
    df = pd.DataFrame([input_dict])
    pred_raw = pipeline.predict(df)[0]
    result = {'prediction': le.inverse_transform([pred_raw])[0] if le else float(pred_raw)}

    if meta['problem_type'] != 'regression':
        try:
            proba = pipeline.predict_proba(df)[0]
            classes = le.classes_.tolist() if le else [str(i) for i in range(len(proba))]
            result['probabilities'] = {str(c): round(float(p), 4) for c,p in zip(classes,proba)}
        except Exception:
            pass
    return result

def predict_batch(session_id, df):
    pipeline, le, meta = load_session(session_id)
    preds_raw = pipeline.predict(df)
    preds = le.inverse_transform(preds_raw) if le else preds_raw.astype(str)

    rows = []
    try:
        probas = pipeline.predict_proba(df)
        classes = le.classes_.tolist() if le else [str(i) for i in range(probas.shape[1])]
        for i, (pred, proba) in enumerate(zip(preds,probas)):
            rows.append({
                'row': i + 1,
                'prediction': str(pred),
                'confidence': round(float(max(proba)), 4),
                'probabilities': {str(c): round(float(p), 4) for c,p in zip(classes, proba)}
            })
    except Exception:
        for i, pred in enumerate(preds):
            rows.append({'row': i + 1, 'prediction': str(pred)})
    return rows
