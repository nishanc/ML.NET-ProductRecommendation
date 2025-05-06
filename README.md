# 🧠 ML.NET Product Recommendation Demo – Co-Purchase Scenario

This demo accompanies the tech talk **"Building Intelligent Applications with ML.NET"**, where we explore how to use **Matrix Factorization** to build a **product recommendation system** based on co-purchase behavior.

---

## 📌 Scenario

Imagine you're running an e-commerce platform. You have customer purchase data and want to recommend products that are **frequently bought together** — even if the user hasn't seen them before.

This solution uses:
- ML.NET’s `MatrixFactorizationTrainer`
- A small sample dataset of `user_id`, `product_id`, and `Label`

---

## 💡 What You'll Learn

- Basics of Matrix Factorization for collaborative filtering
- How to implement recommendations in ML.NET
- Training a model using implicit purchase data
- Making product predictions for a given customer

---

## 🔧 Prerequisites

- [.NET 8 SDK](https://dotnet.microsoft.com/en-us/download)
- Visual Studio or VS Code
- ML.NET NuGet Package:

```
dotnet add package Microsoft.ML
```

## 🛠️ How to Run the Demo

1. **Clone or download the repo**.
2. Open the folder in Visual Studio or run via CLI.
3. Make sure the dataset `amazon.csv` is in the `Data/` folder.
4. Run the program:
 ```bash
 dotnet run