import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载保存的SVM模型
svm_data = joblib.load('SVM.pkl')
model = svm_data['model']
best_threshold = svm_data['best_threshold']

# 加载标准化器
scaler = joblib.load('scaler.pkl')
# 特征范围定义（根据提供的特征范围和数据类型）
feature_ranges = {
    "Hypertension": {"type": "categorical", "options": [0, 1]},
    "BMI": {"type": "numerical", "min": 15.81, "max": 42.97, "default": 21.13},
    "DBIL": {"type": "numerical", "min": 1.1, "max": 598.8, "default": 16.75},
    "Creatinine": {"type": "numerical", "min": 33, "max": 342, "default": 83.00},
    "Hb": {"type": "numerical", "min": 25, "max": 174, "default": 83.99},
    "UONHP": {"type": "numerical", "min": 20, "max": 4750, "default": 1040.00},
    "UO": {"type": "numerical", "min": 0, "max": 16220, "default": 1580.00},
    "VIS": {"type": "numerical", "min": -2, "max": 28, "default": 10},
    "OT": {"type": "numerical", "min":5, "max": 19.5, "default":8.48},
    "Plasma": {"type": "numerical", "min": 0, "max": 4210, "default": 941.42}
}

# Streamlit 界面
st.title("SVM Model with SHAP Visualization")

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

## 预测与 SHAP 可视化
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
    # 创建背景数据（使用特征默认值，代表数据分布）
    background_defaults = [
    props["default"] if props["type"] == "numerical" else props["options"][0]
    for props in feature_ranges.values()
 ]
    background_data = np.array([background_defaults])
    background_scaled = scaler.transform(background_data)

    # 使用 KernelExplainer 替代 TreeExplainer 以支持 SVM 模型
    explainer = shap.KernelExplainer(model.predict_proba, background_scaled)
    shap_df = pd.DataFrame(features_scaled, columns=feature_ranges.keys())
    shap_values = explainer.shap_values(shap_df.values)  # 传递 numpy 数组而非 DataFrame

    # 生成 SHAP 力图
    class_index = predicted_class  # 当前预测类别

    # 检查SHAP值结构并提取正确维度（适配三维数组格式）
    if shap_values.ndim == 3:
        # 三维结构: (样本数, 特征数, 类别数)，取第一个样本和指定类别
        shap_values_to_use = shap_values[0, :, class_index]
    elif isinstance(shap_values, list) and len(shap_values) > class_index:
        # 多类别列表情况
        shap_values_to_use = shap_values[class_index]
    elif shap_values.ndim == 2:
        # 二维数组情况: (样本数, 特征数)
        shap_values_to_use = shap_values[0]
    else:
        shap_values_to_use = shap_values

    # 提取对应类别的预期值
    if isinstance(explainer.expected_value, list) and len(explainer.expected_value) > class_index:
        expected_value_to_use = explainer.expected_value[class_index]
    elif explainer.expected_value.ndim > 0 and explainer.expected_value.shape[0] > class_index:
        expected_value_to_use = explainer.expected_value[class_index]
    else:
        expected_value_to_use = explainer.expected_value

    # 生成并显示SHAP力图
    plt.switch_backend('agg')  # 添加后端设置确保图形正确渲染
    plt.figure(figsize=(16, 8))  # 增大图像尺寸以容纳所有特征
    # 生成并显示SHAP力图
    shap_fig = shap.force_plot(
    expected_value_to_use,
    shap_values_to_use,
    features=shap_df.iloc[0],
    feature_names=list(feature_ranges.keys()),
    matplotlib=True,
    show=False,
)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=600)
    plt.close()
    st.image("shap_force_plot.png")
    