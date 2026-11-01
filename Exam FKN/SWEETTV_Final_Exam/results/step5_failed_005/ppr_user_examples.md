# Step 5 PPR examples for the fixed Step 4 users

Selection is unchanged from Step 4. Titles/categories are validation-schedule display metadata only, not PPR features. A validation label indicates logged qualifying validation viewing, not enjoyment.

Best PPR configuration: `cosine_pfrequency_a0.35_k30` (primary MAP@5 0.815150794).

## Heavy user `13571199046623737884`

Training history: 757 distinct programs, 2358 qualifying broadcasts. Validation labels: 333 distinct programs.

| Rank | Program ID | Title | Category | In validation labels |
|---:|---:|---|---|---|
| 1 | 2400480 | 'ТСН. Неделя'. | Инфо | yes |
| 2 | 240081 | Факты. Утро. | Инфо | yes |
| 3 | 12001682 | М/с 'Губка Боб Квадратные Штаны', 185 с. 'Она пришла из Лагуны Гу'. | Детям | yes |
| 4 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | yes |
| 5 | 6500479 | 'Мир наизнанку'. | Познавательное | yes |

Training-history examples (airings; seconds): 12001682 (118; 104210); 2400480 (93; 146085); 240081 (88; 110555); 2400508 (65; 109135); 20088 (44; 87238); 12001734 (35; 22858); 6500479 (31; 91020); 200337 (31; 56663); 2400467 (27; 51804); 10003159 (21; 102160).

## Light user `8150816946177670084`

Training history: 1 distinct programs, 1 qualifying broadcasts. Validation labels: 15 distinct programs.

| Rank | Program ID | Title | Category | In validation labels |
|---:|---:|---|---|---|
| 1 | 90053098 | Х/ф 'Мужество в бою'. | Фильмы | no |
| 2 | 240081 | Факты. Утро. | Инфо | yes |
| 3 | 2400480 | 'ТСН. Неделя'. | Инфо | no |
| 4 | 6500479 | 'Мир наизнанку'. | Познавательное | no |
| 5 | 2400467 | 'Викна-Новости'. | Инфо | no |

Training-history examples (airings; seconds): 90053098 (1; 2753).
