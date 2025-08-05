# 🛒 Purchase Prediction System

A professional machine learning web application that predicts customer subscription likelihood based on purchase behavior and demographics.

## 🌟 Features

- **Professional Streamlit UI** with modern design and intuitive interface
- **Balanced Machine Learning Model** using Random Forest with 85%+ accuracy
- **Real-time Predictions** with confidence scores and interactive visualizations
- **Comprehensive Analytics** including model performance metrics and data insights
- **Streamlined Input** - Only 6 key features required for prediction

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd PurchasePredictionSystem
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (first time only)
   ```bash
   python train_model.py
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## 📊 Model Performance

- **Accuracy**: 85.2%
- **Precision**: 97.7%
- **Recall**: 72.8%
- **F1-Score**: 83.4%

## 🎯 Input Features

The model uses 6 key customer attributes:

1. **👤 Age** - Customer's age (18-100)
2. **🛍️ Product Category** - Type of product purchased
3. **💰 Purchase Amount** - Amount spent in USD
4. **🌤️ Season** - Season of purchase
5. **⭐ Review Rating** - Customer satisfaction (1-5 stars)
6. **📦 Previous Purchases** - Number of past purchases

## 📁 Project Structure

```
PurchasePredictionSystem/
├── app.py                 # Main Streamlit application
├── train_model.py         # Model training script
├── augment_data.py        # Data augmentation utility
├── pps1.csv              # Dataset (balanced)
├── requirements.txt       # Python dependencies
├── final code.py         # Original model (reference)
├── README.md             # This file
└── .gitignore           # Git ignore rules
```

## 🔧 Technical Details

### Model Architecture
- **Algorithm**: Random Forest Classifier
- **Features**: 6 key customer attributes
- **Balancing**: SMOTE oversampling + class weights
- **Preprocessing**: StandardScaler normalization

### Dataset
- **Size**: 7,900 records (augmented from 3,900)
- **Balance**: 51.4% Subscribe, 48.6% Don't Subscribe
- **Features**: Demographics, purchase behavior, satisfaction metrics

## 🎨 UI Features

- **Interactive Sidebar** - Easy input controls
- **Real-time Predictions** - Instant results with confidence scores
- **Performance Dashboard** - Model metrics and visualizations
- **Data Insights** - Distribution charts and statistics
- **Professional Styling** - Modern gradient design with emojis

## 🛠️ Development

### Adding New Features
1. Modify `train_model.py` for model changes
2. Update `app.py` for UI enhancements
3. Retrain model: `python train_model.py`
4. Test: `streamlit run app.py`

### Data Augmentation
Run `python augment_data.py` to generate additional balanced training data.

## 📈 Model Training Process

1. **Data Loading** - Reads CSV dataset
2. **Feature Selection** - Uses 6 key attributes
3. **Encoding** - Label encoding for categories
4. **Balancing** - SMOTE oversampling
5. **Scaling** - StandardScaler normalization
6. **Training** - Random Forest with balanced weights
7. **Evaluation** - Comprehensive metrics
8. **Saving** - Serialized model and preprocessors

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🎯 Future Enhancements

- [ ] Add more sophisticated feature engineering
- [ ] Implement ensemble methods
- [ ] Add A/B testing capabilities
- [ ] Include customer segmentation
- [ ] Deploy to cloud platforms

## 📞 Support

For questions or issues, please open a GitHub issue or contact the development team.

---

**Built with ❤️ using Python, Streamlit, and Scikit-learn**
