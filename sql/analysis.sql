-- 1. Overall customer count
SELECT COUNT(*) AS total_customers
FROM customers;

-- 2. Overall churn rate
SELECT
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)
        AS churned_customers,
    ROUND(
        100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)
        / COUNT(*), 2
    ) AS churn_rate_pct
FROM customers;

-- 3. Churn rate by contract
SELECT
    Contract,
    COUNT(*) AS customers,
    ROUND(
        100.0 * AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END),
        2
    ) AS churn_rate_pct
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;

-- 4. Average monthly charges by churn status
SELECT
    Churn,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges
FROM customers
GROUP BY Churn;