
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx   # ✅ for network graph
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# 2. Load dataset
df = pd.read_csv(r'C:\Users\Jeevadharan\Documents\ML\24BAD046-ML_Exp8\Groceries_dataset.csv')

# 3. Check columns
print("Columns:", df.columns)

# 4. Convert into transactions
transactions = df.groupby('Member_number')['itemDescription'].apply(list).tolist()

# 5. One-hot encoding
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

# 6. Generate frequent itemsets
frequent_itemsets = apriori(df_encoded, min_support=0.02, use_colnames=True)

# 7. Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)

# 8. Filter rules (lift > 1)
rules = rules[rules['lift'] > 1]

print("\nAssociation Rules:\n")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# ------------------ VISUALIZATION ------------------ #

# 1. Bar chart of frequent itemsets
frequent_itemsets.sort_values('support', ascending=False).head(10).plot(
    x='itemsets', y='support', kind='bar', legend=False
)
plt.title('Top Frequent Itemsets')
plt.xlabel('Itemsets')
plt.ylabel('Support')
plt.xticks(rotation=45)
plt.show()

# 2. Support vs Confidence plot
plt.scatter(rules['support'], rules['confidence'])
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Support vs Confidence')
plt.show()

# 3. Network graph of association rules
# (limit top rules to avoid messy graph)
top_rules = rules.sort_values(by='lift', ascending=False).head(10)

G = nx.DiGraph()

for _, row in top_rules.iterrows():
    for ant in row['antecedents']:
        for con in row['consequents']:
            G.add_edge(ant, con, weight=row['lift'])

plt.figure(figsize=(8,6))
pos = nx.spring_layout(G)

nx.draw(G, pos, with_labels=True, node_size=2000, font_size=8)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title("Association Rules Network Graph")
plt.show()