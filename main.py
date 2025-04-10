import gc
import os
import time
import lightgbm as lgb
from lightgbm.callback import early_stopping
from catboost import CatBoostClassifier
import xgboost as xgb
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import glob
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import pickle

def load_data(data_type):
    """데이터 파일 로드 함수"""
    base_path = f"open/{data_type}"
    categories = {
        "customer": "1",
        "credit": "2",
        "sales": "3",
        "billing": "4",
        "balance": "5",
        "channel": "6",
        "marketing": "7",
        "performance": "8"
    }
    dfs = {}
    for name, prefix in categories.items():
        key = f"{name}_{data_type}_df"
        path = f"{base_path}/{prefix}.*/*.parquet"
        files = sorted(glob.glob(path))
        if not files:
            print(f"{key}: {path} 경로에 파일이 없습니다.")
            continue
        dfs[key] = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        print(f"{key} 로드 완료: {dfs[key].shape}")
    return dfs


def merge_dataframes(base_df, merge_keys, merge_items, prefix="train", dfs=None):
    """데이터프레임 병합 함수"""
    for df_name, step in merge_items:
        full_name = f"{df_name}_{prefix}_df"
        if full_name in dfs:
            print(f"{step} 병합 중: {full_name}")
            base_df = base_df.merge(dfs[full_name], on=merge_keys, how='left')
            print(f"{step} 병합 완료: shape → {base_df.shape}")
            del dfs[full_name]
            gc.collect()
        else:
            print(f"{full_name} 없음 - 병합 생략")
    return base_df


def convert_text_to_int(df, exclude_cols=['ID', '기준년월']):
    """텍스트 데이터를 숫자로 변환하는 함수"""
    object_cols = df.select_dtypes(include='object').columns
    target_cols = [col for col in object_cols if col not in exclude_cols and col != 'Segment']

    if not target_cols:
        return df

    for col in target_cols:
        try:
            # 숫자 추출 패턴 개선: 숫자만 있는 경우와 숫자+단위가 있는 경우 모두 처리
            num_series = df[col].astype(str).str.extract(r'(\d+)(?:개|대|회|이상|회이상|개이상|대이상)?')[0]
            num_series = pd.to_numeric(num_series, errors='coerce')

            # 숫자 변환에 성공한 경우에만 적용
            if num_series.notna().sum() > 0:
                # 변환 실패한 경우는 중앙값으로 대체
                median_val = num_series[num_series.notna()].median()
                num_series = num_series.fillna(median_val)
                df[col] = num_series

            # 변환 실패한 컬럼은 범주형으로 처리
            else:
                print(f"경고: {col} 컬럼은 숫자 변환에 실패하여 범주형으로 처리됩니다.")
        except:
            print(f"경고: {col} 컬럼 변환 중 오류 발생")

    return df


def reduce_mem_usage(df, verbose=True):
    """데이터프레임의 메모리 사용량을 줄이는 함수"""
    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in df.columns:
        if col in ['ID', 'Segment']:  # ID와 타겟 컬럼은 처리하지 않음
            continue

        col_type = df[col].dtype

        if col_type != object:  # 수치형 컬럼
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    reduction = 100 * (start_mem - end_mem) / start_mem

    if verbose:
        print(f"메모리 사용량: {start_mem:.2f} MB → {end_mem:.2f} MB ({reduction:.1f}% 감소)")

    return df


def merge_dataframes(base_df, merge_keys, merge_items, prefix="train", dfs=None):
    """데이터프레임 병합 함수"""
    for df_name, step in merge_items:
        full_name = f"{df_name}_{prefix}_df"
        if full_name in dfs:
            print(f"{step} 병합 중: {full_name}")
            base_df = base_df.merge(dfs[full_name], on=merge_keys, how='left')
            print(f"{step} 병합 완료: shape → {base_df.shape}")
            del dfs[full_name]
            gc.collect()
        else:
            print(f"{full_name} 없음 - 병합 생략")
    return base_df


def convert_text_to_int(df, exclude_cols=['ID', '기준년월']):
    """텍스트 데이터를 숫자로 변환하는 함수"""
    object_cols = df.select_dtypes(include='object').columns
    target_cols = [col for col in object_cols if col not in exclude_cols and col != 'Segment']

    if not target_cols:
        return df

    for col in target_cols:
        try:
            # 숫자 추출 패턴 개선: 숫자만 있는 경우와 숫자+단위가 있는 경우 모두 처리
            num_series = df[col].astype(str).str.extract(r'(\d+)(?:개|대|회|이상|회이상|개이상|대이상)?')[0]
            num_series = pd.to_numeric(num_series, errors='coerce')

            # 숫자 변환에 성공한 경우에만 적용
            if num_series.notna().sum() > 0:
                # 변환 실패한 경우는 중앙값으로 대체
                median_val = num_series[num_series.notna()].median()
                num_series = num_series.fillna(median_val)
                df[col] = num_series

            # 변환 실패한 컬럼은 범주형으로 처리
            else:
                print(f"경고: {col} 컬럼은 숫자 변환에 실패하여 범주형으로 처리됩니다.")
        except:
            print(f"경고: {col} 컬럼 변환 중 오류 발생")

    return df


def reduce_mem_usage(df, verbose=True):
    """데이터프레임의 메모리 사용량을 줄이는 함수"""
    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in df.columns:
        if col in ['ID', 'Segment']:  # ID와 타겟 컬럼은 처리하지 않음
            continue

        col_type = df[col].dtype

        if col_type != object:  # 수치형 컬럼
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    reduction = 100 * (start_mem - end_mem) / start_mem

    if verbose:
        print(f"메모리 사용량: {start_mem:.2f} MB → {end_mem:.2f} MB ({reduction:.1f}% 감소)")

    return df


def handle_numeric_features(train_df, test_df):
    """수치형 변수 처리 함수"""
    # 수치형 변수 식별
    numeric_cols = train_df.select_dtypes(include=['int', 'float']).columns.tolist()

    # 결측치 및 이상값 처리
    for col in numeric_cols:
        # 특수 값 처리 (비즈니스 로직에 따라 조정 필요)
        replace_values = [10101, -999999, -99, 999, 99999999]
        train_df[col] = train_df[col].replace(replace_values, np.nan)
        test_df[col] = test_df[col].replace(replace_values, np.nan)

        # 훈련 데이터 기준으로 이상치 임계값 계산 (25%, 75% 분위수로 변경)
        q1 = train_df[col].quantile(0.25)
        q3 = train_df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # 이상치 클리핑
        orig_dtype = train_df[col].dtype
        train_df[col] = train_df[col].clip(lower=lower_bound, upper=upper_bound).astype(orig_dtype)
        test_df[col] = test_df[col].clip(lower=lower_bound, upper=upper_bound).astype(orig_dtype)

        # 결측치 처리 - 중앙값으로 대체
        median_value = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_value)
        test_df[col] = test_df[col].fillna(median_value)

    return train_df, test_df


def drop_useless_features(train_df, test_df):
    """불필요한 피처 제거 함수"""
    # 상수 컬럼 식별 (훈련 데이터 기준)
    constant_cols = [col for col in train_df.columns
                     if train_df[col].nunique(dropna=False) <= 1 or
                     (col != 'Segment' and
                      train_df[col].dtype in [np.int64, np.float64] and
                      train_df[col].std() < 1e-6)]

    domain_knowledge_drop = [
        # 여기에 도메인 지식을 바탕으로 제거하려는 컬럼 이름을 문자열로 추가
        '남녀구분코드', '가입통신회사코드',
        '이용가능카드수_체크', '이용가능카드수_체크_가족',
        '수신거부여부_TM', '수신거부여부_DM', '수신거부여부_메일', '수신거부여부_SMS',
        '마케팅동의여부', '최종탈회경과일',
        '할인금_기본연회_BOM', 'Life_Stage', '최종카드발경과월',
        '_1순위쇼핑업종', '_1순위쇼핑업종_이용금액',
        '_2순위쇼핑업종', '_2순위쇼핑업종_이용금액',
        '_3순위쇼핑업종', '_3순위쇼핑업종_이용금액',
        '_1순위교통업종', '_1순위교통업종_이용금액',
        '_2순위교통업종', '_2순위교통업종_이용금액',
        '_3순위교통업종', '_3순위교통업종_이용금액',
        '_1순위여유업종', '_1순위여유업종_이용금액',
        '_2순위여유업종', '_2순위여유업종_이용금액',
        '_3순위여유업종', '_3순위여유업종_이용금액',
        '_1순위납부업종', '_1순위납부업종_이용금액',
        '_2순위납부업종', '_2순위납부업종_이용금액',
        '_3순위납부업종', '_3순위납부업종_이용금액',
        '자발한도감액횟수_R12M', '자발한도감액금액_R12M', '자발한감액후경과월',
        '최종카드론_금융상환방식코드',
        '최종카드론_신청경로코드',
        '최종카드론_대출일자'
        '대표결제일',
        '포인트_마일리_환산_BOM',
        '인입횟수_ARS_R6M',
        '이용메건수_ARS_R6M',
        '컨택건수_CA_당사앱_R6M',
        '캠페인접촉건수_R12M', '캠페인접촉일수_R12M',
        '대표결제방법코드', '대표청구지고객주소구분코드',
        '입회경과개월수_신용', '수신거부여부_메일',
        '이용가능카드수_체크', '연체일자_B0M', 'OS구분코드',
    ]

    # 실제로 데이터에 존재하는 컬럼만 필터링
    domain_knowledge_drop = [col for col in domain_knowledge_drop if col in train_df.columns]

    # 모든 제거 대상 컬럼 통합
    all_cols_to_drop = list(set(constant_cols + domain_knowledge_drop))

    if all_cols_to_drop:
        print(f"제거할 피처 총 {len(all_cols_to_drop)}개")
        print(f"- 상수/저분산 컬럼 ({len(constant_cols)}개): {constant_cols}")
        print(f"- 도메인 지식 기반 제거 ({len(domain_knowledge_drop)}개): {domain_knowledge_drop}")

        train_df = train_df.drop(columns=all_cols_to_drop)
        test_df = test_df.drop(columns=[col for col in all_cols_to_drop if col in test_df.columns])

    return train_df, test_df


def drop_highly_correlated_features(train_df, test_df, threshold=0.95):
    """상관관계 높은 피처 제거 함수"""
    # 수치형 컬럼만 선택
    numeric_cols = train_df.select_dtypes(include=['int', 'float']).columns

    # 상관관계 계산
    if len(numeric_cols) > 1:  # 적어도 2개 이상의 수치형 컬럼이 있어야 함
        corr_matrix = train_df[numeric_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        if to_drop:
            print(f"💡 상관관계 {threshold} 초과 피처 제거: {to_drop}")
            train_df = train_df.drop(columns=to_drop)
            test_df = test_df.drop(columns=[col for col in to_drop if col in test_df.columns])

    return train_df, test_df


def handle_missing_values(X, X_test):
    """결측치 처리 함수"""
    for col in X.columns:
        if X[col].isna().sum() > 0:
            if X[col].dtype.kind in 'ifc':  # 수치형 변수면 중앙값으로 대체
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
                X_test[col] = X_test[col].fillna(median_val)
            else:  # 범주형 변수면 최빈값으로 대체
                mode_val = X[col].mode()[0]
                X[col] = X[col].fillna(mode_val)
                X_test[col] = X_test[col].fillna(mode_val)

    return X, X_test


def encode_categorical_features(X, X_test, categorical_features):
    """범주형 변수 인코딩 함수"""
    encoders = {}

    for col in categorical_features:
        if col not in X.columns or col not in X_test.columns:
            print(f"경고: {col} 컬럼이 데이터프레임에 없습니다.")
            continue

        le = LabelEncoder()
        # 훈련 데이터와 테스트 데이터의 모든 카테고리를 합쳐서 인코딩
        all_categories = pd.concat([X[col].astype(str), X_test[col].astype(str)]).unique()
        le.fit(all_categories)

        X[col] = le.transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        encoders[col] = le

    return X, X_test, encoders


def preprocess_data(reload_from_parquet=True):
    """데이터 전처리를 수행하는 함수

    Args:
        reload_from_parquet (bool): True일 경우 이미 전처리된 Parquet 파일이 있으면 로드, 없으면 처음부터 전처리

    Returns:
        tuple: (train_df, test_df) 전처리된 훈련 및 테스트 데이터프레임
    """
    # 이미 전처리된 파일이 있는지 확인
    preprocessed_train_path = "preprocessed_train_data.parquet"
    preprocessed_test_path = "preprocessed_test_data.parquet"

    if reload_from_parquet and os.path.exists(preprocessed_train_path) and os.path.exists(preprocessed_test_path):
        print(f"이미 전처리된 파일 발견: {preprocessed_train_path}, {preprocessed_test_path}")
        print("전처리된 파일을 로드합니다...")
        train_df = pd.read_parquet(preprocessed_train_path)
        test_df = pd.read_parquet(preprocessed_test_path)
        print(f"전처리된 데이터 로드 완료! 훈련 데이터: {train_df.shape}, 테스트 데이터: {test_df.shape}")
        return train_df, test_df

    print("전처리된 파일이 없거나 재전처리가 요청되어 처음부터 전처리를 시작합니다...")

    # 원본 데이터 로드
    print("데이터 로딩 시작...")
    train_dfs = load_data("train")
    test_dfs = load_data("test")
    print("데이터 로딩 완료!")

    merge_list = [
        ("sales", "Step2"),
        ("billing", "Step3"),
        ("balance", "Step4"),
        ("channel", "Step5"),
        ("marketing", "Step6"),
        ("performance", "최종")
    ]

    print("데이터 병합 시작...")
    train_df = train_dfs["customer_train_df"].merge(train_dfs["credit_train_df"], on=["기준년월", "ID"], how="left")
    test_df = test_dfs["customer_test_df"].merge(test_dfs["credit_test_df"], on=["기준년월", "ID"], how="left")

    train_df = merge_dataframes(train_df, ["기준년월", "ID"], merge_list, prefix="train", dfs=train_dfs)
    test_df = merge_dataframes(test_df, ["기준년월", "ID"], merge_list, prefix="test", dfs=test_dfs)
    print("데이터 병합 완료!")

    print("메모리 사용량 최적화 중...")
    train_df = reduce_mem_usage(train_df)
    test_df = reduce_mem_usage(test_df)

    # 병합된 데이터 Parquet으로 저장 (중간 결과)
    print("병합된 데이터 Parquet 파일로 저장 중...")
    train_df.to_parquet("merged_train_data.parquet", index=False)
    test_df.to_parquet("merged_test_data.parquet", index=False)
    print("병합된 데이터 저장 완료! 파일명: merged_train_data.parquet, merged_test_data.parquet")

    print("텍스트를 숫자로 변환 중...")
    train_df = convert_text_to_int(train_df)
    test_df = convert_text_to_int(test_df)

    print("수치형 데이터 전처리 중...")
    train_df, test_df = handle_numeric_features(train_df, test_df)

    print("불필요한 피처 제거 중...")
    train_df, test_df = drop_useless_features(train_df, test_df)

    print("상관관계 높은 피처 제거 중...")
    train_df, test_df = drop_highly_correlated_features(train_df, test_df)

    # 전처리 완료된 데이터 저장
    print("전처리된 데이터 Parquet 파일로 저장 중...")
    train_df.to_parquet(preprocessed_train_path, index=False)
    test_df.to_parquet(preprocessed_test_path, index=False)
    print(f"전처리된 데이터 저장 완료! 파일명: {preprocessed_train_path}, {preprocessed_test_path}")

    print(f"전처리 완료! 훈련 데이터: {train_df.shape}, 테스트 데이터: {test_df.shape}")
    return train_df, test_df


def scale_features(X_train, X_test, scaler=None):
    """수치형 특성 스케일링"""
    numeric_cols = X_train.select_dtypes(include=['int', 'float']).columns

    if scaler is None:
        scaler = StandardScaler()
        fit_scaler = True
    else:
        fit_scaler = False

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    if fit_scaler:
        X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    else:
        X_train_scaled[numeric_cols] = scaler.transform(X_train[numeric_cols])

    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train_scaled, X_test_scaled, scaler


def train_and_evaluate_models(X, y, X_test, test_ids, test_df, n_folds=5, random_state=42):
    """모델 학습 및 교차 검증을 수행하는 함수"""
    # 전체 학습 시작 시간 기록
    total_start_time = time.time()

    # 교차 검증 설정
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # 각 모델별 OOF 예측 결과
    oof_preds = {
        'lightgbm': np.zeros((X.shape[0], len(np.unique(y)))),
        'xgboost': np.zeros((X.shape[0], len(np.unique(y)))),
        'catboost': np.zeros((X.shape[0], len(np.unique(y))))
    }

    # 테스트 데이터에 대한 예측 결과
    test_preds = {
        'lightgbm': np.zeros((X_test.shape[0], len(np.unique(y)))),
        'xgboost': np.zeros((X_test.shape[0], len(np.unique(y)))),
        'catboost': np.zeros((X_test.shape[0], len(np.unique(y))))
    }

    # 각 모델별 총 학습 시간
    model_times = {
        'lightgbm': 0,
        'xgboost': 0,
        'catboost': 0
    }

    # Label Encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"학습 시작: {n_folds}겹 교차 검증")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_encoded)):
        fold_start_time = time.time()
        print(f"\nFold {fold + 1}/{n_folds}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

        # 스케일링 적용
        X_train_scaled, X_val_scaled, scaler = scale_features(X_train, X_val)
        # 테스트 데이터도 같은 스케일러로 변환
        _, X_test_scaled, _ = scale_features(X_test, X_test, scaler=scaler)

        # LightGBM
        print("LightGBM 모델 학습 중...")
        lgb_start = time.time()

        # 하이퍼파라미터 미세 조정
        lgb_params = {
            'objective': 'multiclass',
            'num_class': len(np.unique(y)),
            'boosting_type': 'gbdt',
            'n_estimators': 2000,  # 증가
            'learning_rate': 0.005,  # 감소
            'max_depth': 6,  # 조정
            'num_leaves': 31,  # 조정
            'min_child_samples': 20,
            'min_child_weight': 0.001,
            'subsample': 0.85,  # 조정
            'colsample_bytree': 0.85,  # 조정
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': random_state
        }

        lgb_model = lgb.LGBMClassifier(**lgb_params)

        lgb_model.fit(X_train_scaled, y_train)

        lgb_end = time.time()
        lgb_time = lgb_end - lgb_start
        model_times['lightgbm'] += lgb_time
        print(f"LightGBM 모델 학습 완료: {lgb_time:.2f}초")

        oof_preds['lightgbm'][val_idx] = lgb_model.predict_proba(X_val_scaled)
        test_preds['lightgbm'] += lgb_model.predict_proba(X_test_scaled) / n_folds

        # XGBoost
        print("XGBoost 모델 학습 중...")
        xgb_start = time.time()

        # 하이퍼파라미터 미세 조정
        xgb_params = {
            'objective': 'multi:softprob',
            'num_class': len(np.unique(y)),
            'n_estimators': 2000,  # 증가
            'learning_rate': 0.005,  # 감소
            'max_depth': 7,  # 증가
            'min_child_weight': 3,  # 추가
            'subsample': 0.85,  # 조정
            'colsample_bytree': 0.85,  # 조정
            'gamma': 0.1,  # 추가
            'reg_alpha': 0.1,  # 추가
            'reg_lambda': 1.0,  # 추가
            'random_state': random_state
        }
        xgb_model = xgb.XGBClassifier(**xgb_params)

        xgb_model.fit(X_train_scaled, y_train)

        xgb_end = time.time()
        xgb_time = xgb_end - xgb_start
        model_times['xgboost'] += xgb_time
        print(f"XGBoost 모델 학습 완료: {xgb_time:.2f}초")

        oof_preds['xgboost'][val_idx] = xgb_model.predict_proba(X_val_scaled)
        test_preds['xgboost'] += xgb_model.predict_proba(X_test_scaled) / n_folds

        # CatBoost
        print("CatBoost 모델 학습 중...")
        cat_start = time.time()

        # 하이퍼파라미터 미세 조정
        cat_params = {
            'iterations': 2000,  # 증가
            'learning_rate': 0.005,  # 감소
            'depth': 8,  # 증가
            'l2_leaf_reg': 5,  # 조정
            'random_seed': random_state,
            'verbose': False,
            'bootstrap_type': 'Bernoulli',  # 추가
            'subsample': 0.85,  # 추가
        }
        cat_model = CatBoostClassifier(**cat_params)

        cat_model.fit(X_train_scaled, y_train)

        cat_end = time.time()
        cat_time = cat_end - cat_start
        model_times['catboost'] += cat_time
        print(f"CatBoost 모델 학습 완료: {cat_time:.2f}초")

        oof_preds['catboost'][val_idx] = cat_model.predict_proba(X_val_scaled)
        test_preds['catboost'] += cat_model.predict_proba(X_test_scaled) / n_folds

        fold_end_time = time.time()
        print(f"Fold {fold + 1} 총 소요 시간: {fold_end_time - fold_start_time:.2f}초")

    # 각 모델별 OOF 성능 평가
    model_accuracies = {}
    for model_name in oof_preds:
        oof_pred_labels = np.argmax(oof_preds[model_name], axis=1)
        oof_accuracy = accuracy_score(y_encoded, oof_pred_labels)
        model_accuracies[model_name] = oof_accuracy
        print(f"{model_name} OOF 정확도: {oof_accuracy:.4f}, 총 학습 시간: {model_times[model_name]:.2f}초")

    # 앙상블 - 평균
    ensemble_oof_preds = (oof_preds['lightgbm'] + oof_preds['xgboost'] + oof_preds['catboost']) / 3
    ensemble_oof_pred_labels = np.argmax(ensemble_oof_preds, axis=1)
    ensemble_accuracy = accuracy_score(y_encoded, ensemble_oof_pred_labels)
    print(f"앙상블 OOF 정확도: {ensemble_accuracy:.4f}")

    # 각 모델의 성능에 기반한 가중치 계산
    total_accuracy = sum(model_accuracies.values())
    model_weights = {model: acc / total_accuracy for model, acc in model_accuracies.items()}

    print("모델 가중치:")
    for model_name, weight in model_weights.items():
        print(f"  - {model_name}: {weight:.4f}")

    # 가중치 기반 앙상블
    weighted_ensemble_test_preds = (
            model_weights['lightgbm'] * test_preds['lightgbm'] +
            model_weights['xgboost'] * test_preds['xgboost'] +
            model_weights['catboost'] * test_preds['catboost']
    )

    # 스태킹 앙상블 모델 학습
    print("스태킹 앙상블 모델 학습 중...")
    stacking_start = time.time()

    # 메타 모델 학습을 위한 OOF 예측값 준비
    meta_train = np.column_stack([
        oof_preds['lightgbm'].reshape(X.shape[0], -1),
        oof_preds['xgboost'].reshape(X.shape[0], -1),
        oof_preds['catboost'].reshape(X.shape[0], -1)
    ])

    # 테스트 데이터 예측값 준비
    meta_test = np.column_stack([
        test_preds['lightgbm'].reshape(X_test.shape[0], -1),
        test_preds['xgboost'].reshape(X_test.shape[0], -1),
        test_preds['catboost'].reshape(X_test.shape[0], -1)
    ])

    # 메타 모델 학습
    from sklearn.linear_model import LogisticRegression
    meta_model = LogisticRegression(max_iter=1000, multi_class='multinomial', random_state=random_state)
    meta_model.fit(meta_train, y_encoded)

    # 스태킹 모델의 테스트 데이터 예측
    stacking_test_preds = meta_model.predict_proba(meta_test)

    stacking_end = time.time()
    stacking_time = stacking_end - stacking_start
    print(f"스태킹 앙상블 모델 학습 완료: {stacking_time:.2f}초")

    # 최종 앙상블: 가중치 앙상블과 스태킹 앙상블 조합 (0.6:0.4 비율)
    final_ensemble_preds = 0.6 * weighted_ensemble_test_preds + 0.4 * stacking_test_preds
    final_ensemble_pred_labels = np.argmax(final_ensemble_preds, axis=1)

    # 역변환하여 원래 레이블로 변환
    final_ensemble_pred_classes = le.inverse_transform(final_ensemble_pred_labels)

    # 테스트 데이터에 예측 결과 추가
    test_data = test_df.copy()
    test_data['pred_label'] = final_ensemble_pred_classes

    # ID 별로 가장 많이 예측된 Segment를 선택하여 제출 파일 생성
    submission = test_data.groupby("ID")["pred_label"] \
        .agg(lambda x: x.value_counts().idxmax()) \
        .reset_index()
    submission.columns = ["ID", "Segment"]

    # 제출 파일 생성
    submission.to_csv('base_submit.csv', index=False)
    print("제출 파일 생성 완료: base_submit.csv")

    # 모델 저장
    print("최종 모델 저장 중...")
    with open('lgb_model.pkl', 'wb') as f:
        pickle.dump(lgb_model, f)
    with open('xgb_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    with open('cat_model.pkl', 'wb') as f:
        pickle.dump(cat_model, f)
    with open('meta_model.pkl', 'wb') as f:
        pickle.dump(meta_model, f)
    print("모델 저장 완료: lgb_model.pkl, xgb_model.pkl, cat_model.pkl, meta_model.pkl")

    # 전체 학습 종료 시간 및 총 소요 시간
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f"\n전체 모델링 과정 소요 시간: {total_time:.2f}초 ({total_time / 60:.2f}분)")

    return submission, ensemble_accuracy

# 실행 환경 설정
RELOAD_FROM_PARQUET = True  # 전처리된 파일 사용 여부
SKIP_PREPROCESSING = False  # 모델링 단계만 실행 여부

# 파일 경로 정의
FINAL_TRAIN_PATH = "final_train_data.parquet"
FINAL_TEST_PATH = "final_test_data.parquet"
PREPROCESSED_TRAIN_PATH = "preprocessed_train_data.parquet"
PREPROCESSED_TEST_PATH = "preprocessed_test_data.parquet"

# 이미 모델링 준비가 완료된 파일이 있는지 확인
if os.path.exists(FINAL_TRAIN_PATH) and os.path.exists(FINAL_TEST_PATH) and SKIP_PREPROCESSING:
    print(f"모델링 준비가 완료된 파일 발견: {FINAL_TRAIN_PATH}, {FINAL_TEST_PATH}")
    print("모델링용 파일을 로드합니다...")

    train_df = pd.read_parquet(FINAL_TRAIN_PATH)
    X = train_df.drop(columns=["ID", "Segment"])
    y = train_df["Segment"]

    test_df = pd.read_parquet(FINAL_TEST_PATH)
    test_ids = test_df["ID"].copy()
    X_test = test_df.drop(columns=["ID"])

    print(f"모델링용 데이터 로드 완료! 훈련 피처: {X.shape}, 테스트 피처: {X_test.shape}")
else:
    # 데이터 로드 및 전처리
    train_df, test_df = preprocess_data(RELOAD_FROM_PARQUET)

    # 특성과 타깃 분리
    feature_cols = [col for col in train_df.columns if col not in ["ID", "Segment"]]
    X = train_df[feature_cols].copy()
    y = train_df["Segment"].copy()

    # 테스트 데이터 ID 저장
    test_ids = test_df["ID"].copy()
    X_test = test_df[feature_cols].copy()

    # 결측치 처리
    X, X_test = handle_missing_values(X, X_test)

    # 범주형 변수 식별
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # 범주형 변수 인코딩
    if categorical_features:
        print(f"범주형 변수 인코딩 중... {categorical_features}")
        X, X_test, _ = encode_categorical_features(X, X_test, categorical_features)

    # 모델링용 데이터 저장 (특성 엔지니어링 완료된 데이터)
    print("모델링용 데이터 저장 중...")
    final_train = pd.concat([pd.DataFrame({'ID': train_df['ID']}), X, pd.DataFrame({'Segment': y})], axis=1)
    final_test = pd.concat([pd.DataFrame({'ID': test_df['ID']}), X_test], axis=1)

    final_train.to_parquet(FINAL_TRAIN_PATH, index=False)
    final_test.to_parquet(FINAL_TEST_PATH, index=False)
    print(f"모델링용 데이터 저장 완료! 파일명: {FINAL_TRAIN_PATH}, {FINAL_TEST_PATH}")

# 모델 학습 및 평가
submission, accuracy = train_and_evaluate_models(X, y, X_test, test_ids, test_df)
print(f"최종 모델 정확도: {accuracy:.4f}")

# 제출 양식 확인
print(f"제출 파일 크기: {submission.shape}")