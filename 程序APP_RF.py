import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载保存的随机森林模型
rf_data = joblib.load('RF.pkl')
model = rf_data['model']
best_threshold = rf_data['best_threshold']
# 加载标准化器
scaler = joblib.load('scaler.pkl')
# 特征范围定义（根据提供的特征范围和数据类型）
feature_ranges = {
    "Hypertension": {"type": "categorical", "options": [1, 0]},
    "BMI": {"type": "numerical", "min": 15.81, "max": 42.97, "default": 22.54},
    "DBIL": {"type": "numerical", "min": 1.1, "max": 598.8, "default": 45.28},
    "Creatinine": {"type": "numerical", "min": 33, "max": 342, "default": 70.75},
    "Hb": {"type": "numerical", "min": 25, "max": 174, "default": 107.00},
    "UONHP": {"type": "numerical", "min": 20, "max": 4750, "default": 1400.00},
    "UO": {"type": "numerical", "min": 0, "max": 16220, "default": 2260.00},
    "VIS": {"type": "numerical", "min": -2, "max": 28, "default": 4.00},
    "OT": {"type": "numerical", "min":5, "max": 19.5, "default":10.00},
    "Plasma": {"type": "numerical", "min": 0, "max": 4210, "default": 1000.00}
}

# Streamlit 界面
#终端中运行
#e:
#cd 二分类肝移植机器模型\最佳变量特征\Web部署
#streamlit run 程序APP.py


st.title("RandomForest Model with SHAP Visualization")

# 动态生成输入项
st.header("Enter the following feature values:")
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

# 在特征收集后添加
features = np.array([feature_values])
if features.shape[1] != model.n_features_in_:
    st.error(f"特征数量不匹配！模型需要 {model.n_features_in_} 个特征，但输入了 {features.shape[1]} 个")
    st.stop()


if st.button("Predict", key="unique_predict_button"):
    features = np.array([feature_values])
    # 应用标准化器对特征进行处理并保留两位小数
    features_scaled = np.round(scaler.transform(features), decimals=2)
    
    # 模型预测概率
    predicted_proba = model.predict_proba(features_scaled)[0]
    positive_class_prob = predicted_proba[1]  # 假设1是正类
    
    # 使用最佳阈值确定预测类别
    predicted_class = 1 if positive_class_prob >= best_threshold else 0
    probability = positive_class_prob * 100

    # 显示预测结果，使用 Matplotlib 渲染指定字体
    text = f"Based on feature values, predicted possibility of AKI is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # 计算 SHAP 值（使用标准化后的数据）
    explainer = shap.TreeExplainer(model)
    shap_df = pd.DataFrame(np.round(features_scaled, decimals=2), columns=feature_ranges.keys())
    shap_values = explainer.shap_values(shap_df)
    
    # 生成 SHAP 力图
    class_index = predicted_class  # 当前预测类别

    # 从三维SHAP值数组中提取当前样本和类别的SHAP值 (1,10,2) -> (10,)
    if shap_values.ndim == 3:
    # 提取第一个样本、所有特征、当前类别的SHAP值
        shap_values_to_use = shap_values[0, :, class_index]
    else:
    # 兼容其他可能的维度结构
        shap_values_to_use = shap_values[class_index] if isinstance(shap_values, list) else shap_values

    # 提取对应类别的预期值（基准值）
    expected_value_to_use = explainer.expected_value[class_index] if explainer.expected_value.ndim > 0 else explainer.expected_value

    # 生成并显示SHAP力图
    shap_fig = shap.force_plot(
    expected_value_to_use,
    shap_values_to_use,
    shap_df,
    matplotlib=True,
    )
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=300)
    plt.close()
    st.image("shap_force_plot.png")


