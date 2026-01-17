# Human Activity Recognition (HAR) Classification Project

A supervised learning project to classify 6 human activities using sensor data from accelerometers and gyroscopes.

## Project Structure

```
activity-analysis/
├── data/
│   ├── train.csv          # Training dataset
│   └── test.csv           # Test dataset
├── notebooks/
│   └── 01_HAR_Classification_Starter.ipynb  # Starter notebook template
├── src/                   # Source code (optional, for later)
├── HAR_Classification_Guide.md  # Comprehensive step-by-step guide
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Learning

1. **Read the Guide**: Open `HAR_Classification_Guide.md` for detailed explanations of each step
2. **Use the Notebook**: Open `notebooks/01_HAR_Classification_Starter.ipynb` in Jupyter
3. **Write Code**: Follow the TODO comments and write your own code with guidance from the markdown guide

### 3. Workflow

For each step:
1. Read the explanation in `HAR_Classification_Guide.md`
2. Understand the concepts and why each step matters
3. Write code in the notebook following the TODO comments
4. Run and test your code
5. Answer the reflection questions

## Project Goals

- **Learn**: Understand supervised learning workflow
- **Practice**: Write code yourself with guidance
- **Build**: Create a resume-worthy project with clear visualizations
- **Understand**: Learn why each step is important

## Target Activities

The model will classify these 6 activities:
- WALKING
- WALKING_UPSTAIRS
- WALKING_DOWNSTAIRS
- SITTING
- STANDING
- LAYING

## Dataset Information

- **Training set**: 7,354 samples
- **Test set**: 2,949 samples
- **Features**: 561 extracted features from accelerometer and gyroscope signals
- **Metadata**: Subject IDs and activity labels

## Learning Approach

This project is designed for **guided learning**:
- ✅ You write the code
- ✅ I provide explanations and concepts
- ✅ Step-by-step guidance with clear rationale
- ✅ Best practices and industry standards

## Key Concepts You'll Learn

1. **Data Exploration**: Understanding your dataset before modeling
2. **Preprocessing**: Scaling, encoding, handling missing values
3. **Model Selection**: Why Random Forest works well for this problem
4. **Hyperparameter Tuning**: Systematic approach to improving models
5. **Evaluation**: Beyond accuracy - confusion matrices, per-class metrics
6. **Interpretability**: Feature importance and model understanding
7. **Visualization**: Creating publication-ready plots

## Next Steps

1. Start with Step 1 in the guide: Loading and Exploring Data
2. Work through each step sequentially
3. Don't skip ahead - each step builds on the previous one
4. Ask questions if concepts are unclear
5. Experiment and try variations once you understand the basics

## Tips for Success

- **Read the guide first** before writing code
- **Understand the "why"** not just the "how"
- **Experiment** with different approaches
- **Visualize everything** - plots reveal insights
- **Document your choices** - explain why you made certain decisions

## Resources

- **Main Guide**: `HAR_Classification_Guide.md` - Comprehensive explanations
- **Starter Notebook**: `notebooks/01_HAR_Classification_Starter.ipynb` - Code template
- **Dataset**: Kaggle Human Activity Recognition dataset

Good luck! 🚀

