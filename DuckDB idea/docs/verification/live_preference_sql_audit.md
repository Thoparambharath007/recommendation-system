# Actual live SQL audit

Model: `nvidia/nemotron-3.5-lightning-30b-a3b`

The model receives allowed column names, DuckDB types and categorical vocabulary. The app extracts office/comfort preferences before the model call and retains them for ranking. These are captured live responses, not proposed SQL.

## Direct full-query test

a comfortable white shirt for office use

```json
{
  "sql": "SELECT parent_asin FROM shirts WHERE color = 'white'",
  "explanation": "Interpreted request to filter for white shirts for office wear.",
  "sort": "most_reviewed",
  "unsupported": [
    "comfort requirements",
    "office occasion hard filter"
  ]
}
```

## 1. a comfortable white shirt for office use

Sent to model: `a white shirt use`

```sql
SELECT parent_asin FROM shirts WHERE contains('|' || lower(coalesce(color,'')) || '|', '|white|')
```

Model sort: `recommended`. HTTP status: 200.

Matching image groups: 3438. App preferences: `{"office": true, "comfort": true}`.

Matches shirts with white color token in the pipe-delimited color list. White must be the largest shade and cover at least 50% of saved shirt pixels; small accents and shadows do not qualify. Ranked using office-style evidence and comfort wording and fabric/fit evidence from recorded listings; these are suitability signals, not guarantees.

| Product ID | Title | Preference score |
|---|---|---|
| B07QZ2J6DG | Perry Ellis Men's Solid Knit Oxford Short Sleeve Shirt | 11 |
| B07F1ZLZVP | Chaps Men's Dress Shirts Regular Fit Stretch Buttondown Collar Solid | 11 |
| B07991TMQR | Meraki Men's Oxford Casual Short Sleeve Shirt | 11 |

## 2. comfortable black shirt for office under 1200 rupees

Sent to model: `black shirt under 1200 rupees`

```sql
SELECT parent_asin FROM shirts WHERE color = 'black' AND price_inr_est < 1200
```

Model sort: `lowest_price`. HTTP status: 200.

Matching image groups: 73. App preferences: `{"office": true, "comfort": true}`.

Exact black color filter and INR price limit under 1200 Black must be the largest shade and cover at least 50% of saved shirt pixels; small accents and shadows do not qualify. Ranked using office-style evidence and comfort wording and fabric/fit evidence from recorded listings; these are suitability signals, not guarantees.

| Product ID | Title | Preference score |
|---|---|---|
| B0B2WV94SQ | Men's Fashion Shirt Casual Short Sleeve Printed Button Down Shirt Tropical Beach Shirts Vintage Summer Hawaiian | 0 |
| B0BQ3FWQVG | Zeiyignr Valentine's Day Hawaiian Shirt for Men Relaxed-Fit Short Sleeve Button Down Shirts Vacation Dating Heart Print Tops | 6 |
| B07VKT2XWP | Mens Funny Avocado Printed Standard Fit Cotton Button Down Short Sleeve Casual Hawaiian Shirts Beach Top | 7 |

## 3. a white shirt for office work full sleeve

Sent to model: `a white shirt full sleeve`

```sql
SELECT parent_asin FROM shirts WHERE color = 'white' AND sleeve = 'full'
```

Model sort: `recommended`. HTTP status: 200.

Matching image groups: 1544. App preferences: `{"office": true}`.

Filter for white color and full sleeve. White must be the largest shade and cover at least 50% of saved shirt pixels; small accents and shadows do not qualify. Ranked using office-style evidence from recorded listings; these are suitability signals, not guarantees.

| Product ID | Title | Preference score |
|---|---|---|
| B07DPJ6V73 | J.Crew Factory Men's Slim Fit Oxford Cotton Shirt | 7 |
| B016IXRFSW | Urban Boundaries Mens Button Down Point Collar Black White Dress Shirts | 7 |
| B07G2RVV3B | Amazon Brand - Goodthreads Men's "The Perfect Oxford Shirt" Slim-Fit Long-Sleeve Solid Shirt | 7 |

## 4. comfortable blue cotton shirt size M under 1500 rupees

Sent to model: `blue cotton shirt size M under 1500 rupees`

```sql
SELECT parent_asin FROM shirts WHERE contains('|' || lower(coalesce(color,'')) || '|','|blue|') AND sizes LIKE '%|m|%' AND price_inr_est < 1500
```

Model sort: `most_reviewed`. HTTP status: 422.

Error: {"detail": "Use a supported size label such as M, L, XL or XXL"}

## 5. plain pink shirt full sleeve under 1200 rupees

Sent to model: `plain pink shirt full sleeve under 1200 rupees`

```sql
SELECT parent_asin FROM shirts WHERE color = 'pink' AND pattern = 'plain' AND sleeve = 'full' AND price_inr_est < 1200
```

Model sort: `recommended`. HTTP status: 200.

Matching image groups: 1. App preferences: `{}`.

Filters for exact pink color, plain pattern, full sleeve, and price under 1200 INR. Only products tagged solely pink are included; mixed color tags are excluded. Pink must be the largest shade and cover at least 50% of saved shirt pixels; small accents and shadows do not qualify.

| Product ID | Title | Preference score |
|---|---|---|
| B005CHVJJ2 | Authentic Bavarian Trachten Shirt Gingham Checkered White red for Leather Trousers | 0 |

## Findings

- Office/comfort are intentionally ranking preferences, not mandatory keyword filters.
- White office/comfort query used both preferences and returned top scores of 11.
- Black office/comfort budget query incorrectly received lowest_price sort, bypassing preference-first ranking despite the explanation saying it ranked on evidence.
- Cotton + size M query omitted cotton, chose an unrequested sort and produced an invalid size expression; the app rejected it with HTTP 422.
- Direct full query incorrectly marked office and comfort unsupported; the normal app path avoids this by extracting preferences.
- These five examples do not establish overall model reliability. Dataset attribute accuracy is a separate issue.
