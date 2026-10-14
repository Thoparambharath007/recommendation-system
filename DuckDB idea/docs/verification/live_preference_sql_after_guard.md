# Request checks: live verification

Model: `nvidia/nemotron-3.5-lightning-30b-a3b`

Five real-provider requests completed with HTTP 200. Generated SQL is recorded separately from application-enforced SQL. Ranking suitability remains based on listing evidence, not guarantees.

| Query | Recorded matching image groups | App sort | Enforced fields |
|---|---:|---|---|
| a comfortable white shirt for office use | 3438 | recommended | color |
| comfortable black shirt for office under 1200 rupees | 73 | recommended | color, price_inr_est |
| a white shirt for office work full sleeve | 1544 | recommended | color, sleeve |
| comfortable blue cotton shirt size M under 1500 rupees | 0 | recommended | color, fabric, sizes, price_inr_est |
| plain pink shirt full sleeve under 1200 rupees | 1 | recommended | color, sleeve, pattern, price_inr_est |

The blue/cotton/M/budget request previously failed with HTTP 422; now it executes successfully and returns zero recorded matches without weakening the requested filters.

The black office/comfort request previously ranked a Hawaiian listing first after the model chose lowest_price. Now recommended ranking returns the three top listings with preference scores of 11.

## Actual model SQL after the prompt update

a comfortable white shirt for office use

```sql
SELECT parent_asin FROM shirts WHERE contains('|' || lower(coalesce(color,'')) || '|','|white|')
```

comfortable black shirt for office under 1200 rupees

```sql
SELECT parent_asin FROM shirts WHERE contains('|' || lower(coalesce(color,'')) || '|','|black|') AND price_inr_est < 1200
```

a white shirt for office work full sleeve

```sql
SELECT parent_asin FROM shirts WHERE contains('|' || lower(coalesce(color,'')) || '|','|white|') AND sleeve = 'full'
```

comfortable blue cotton shirt size M under 1500 rupees

```sql
SELECT parent_asin FROM shirts WHERE contains('|' || lower(coalesce(color,'')) || '|','|blue|') AND contains(';' || lower(coalesce(fabric,'')) || ';',';cotton;') AND contains('|' || lower(coalesce(sizes,'')) || '|','|m|') AND price_inr_est < 1500
```

plain pink shirt full sleeve under 1200 rupees

```sql
SELECT parent_asin FROM shirts WHERE color = 'pink' AND pattern = 'plain' AND sleeve = 'full' AND price_inr_est < 1200
```

## Limitations

Independent request checks cover common explicit wording for color, material, size, sleeves, patterns, pockets and budgets. They preserve other validated model conditions and reject ambiguous mixed grouping. They do not guarantee interpretation of every possible sentence or correctness of dataset attributes.
