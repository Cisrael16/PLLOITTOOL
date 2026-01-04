# PLL OIT Dashboard (Streamlit) - Multi-Year Edition

## What's New in v6.0 🎉
- **📅 Multi-Year Analysis**: Select any season from 2022-2025 to analyze
- **⚖️ Year-over-Year Comparison**: Compare two seasons side-by-side
- **📈 Player Progression Tracking**: See who improved/declined between seasons
- **🔄 Role Change Detection**: Track players who shifted archetypes year-to-year
- **Plus all v5.1 features**: Team Summary, Percentile Rankings, Player Comparison

## What You Need

### Files to copy to your Mac:
- `app.py` (the new multi-year version)
- `requirements.txt`
- `README.md`

### CSV Data Files (organize these in the same folder):
```
your_folder/
├── app.py
├── requirements.txt
├── PLL_2022_Touch_Rate.csv
├── 2022_pll-player-stats.csv
├── PLL_2023_Touch_Rate.csv
├── 2023_pll-player-stats.csv
├── PLL_2024_Touch_Rate.csv
├── 2024_pll-player-stats.csv
├── PLL_2025_Touch_Rate.csv
└── pll-player-stats.csv (this is your 2025 stats)
```

**Important naming conventions:**
- Touch Rate files: `PLL_YYYY_Touch_Rate.csv` (e.g., `PLL_2024_Touch_Rate.csv`)
- Stats files for historical years: `YYYY_pll-player-stats.csv` (e.g., `2024_pll-player-stats.csv`)
- Stats file for 2025: `pll-player-stats.csv` (no year prefix)

## Install (Mac)
Recommended Python: 3.11 or 3.12.

In Terminal, `cd` into your folder and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```

Streamlit opens at: http://localhost:8501

## Using the Multi-Year Features

### Season Selection
Use the **"Select Season"** dropdown in the sidebar to choose which year to analyze (2022-2025).

### Year Comparison Mode
1. Check **"Enable Year Comparison"** in the sidebar
2. Select a second season from the **"Compare with"** dropdown
3. A new **"Year Comparison"** tab appears showing:
   - **Biggest Improvers**: Players with largest OIT gains
   - **Biggest Decliners**: Players with largest OIT drops
   - **Scatter Plot**: Visual comparison of year-over-year performance
   - **Role Changes**: Players who shifted archetypes (Shooter → Facilitator, etc.)

### Example Use Cases
- **Rookie progression**: Compare 2024 vs 2025 to see how draft picks developed
- **Team evolution**: See how team OIT rankings changed after roster moves
- **Career arcs**: Track individual players across multiple seasons
- **Meta analysis**: Identify if league-wide trends favor certain roles

## OIT Calculation
```
OIT = (Goal Rate × Goal Value) + (Assist Rate × Assist Weight) - (Turnover Rate × Turnover Penalty)
```

- **Goal Value**: Points per goal (calculated from league data for that season)
- **Assist Weight**: Adjustable (default 0.7)
- **Turnover Penalty**: Adjustable (default 0.6)
- **OIT Index**: Player OIT relative to that season's average (100 = average)

## All Features

### Tab 1: Player Analysis
- Usage vs Efficiency scatter plot
- Player comparison mode
- Usage optimization (should get more/fewer touches)
- Top OIT rankings

### Tab 2: Team Summary
- Team OIT rankings bar chart
- Team statistics comparison
- Role distribution by team

### Tab 3: Percentile Rankings
- Individual player percentile charts
- Full percentile rankings table
- See where players rank across all metrics

### Tab 4: Year Comparison (when enabled)
- Biggest improvers/decliners
- Year-over-year scatter plot
- Role change tracking

## Filters & Controls
- **Season selector**: Choose year to analyze
- **Min touches**: Set minimum touches to qualify
- **Filter by team**: Focus on specific teams
- **Assist weight**: Adjust playmaking value (0.0-2.0)
- **Turnover penalty**: Adjust turnover cost (0.0-2.0)

## Player Roles
Automatically categorized based on statistical profile:
- **Shooter**: High shot rate, lower assist rate
- **Facilitator**: High assist rate with good ball movement
- **Efficient Finisher**: Elite goal conversion on lower usage
- **Ball-Dominant Creator**: High usage with strong playmaking
- **Risky Creator**: Creates offense but with higher turnovers
- **Other**: Balanced profile

## Troubleshooting

### "Missing data files for YYYY" error
- Check that your CSV files follow the exact naming convention above
- Make sure files are in the same folder as `app.py`
- Verify you have both touch rate AND stats files for each year

### "No players with X+ touches found in both years"
- Lower the "Min touches" filter
- Some years may have different qualifying thresholds
- Check that player names match exactly across CSVs

### Data format issues
- Touch Rate CSVs should have alternating team logo/player name rows
- Stats CSVs should have `goals` and `scoringPoints` columns
- All data should be numeric (not text)

### Want to add more years?
Edit the `YEARS_CONFIG` dictionary in `app.py` to add years 2019-2021 if you have the touch rate data for those seasons.

## Tips for Best Results

**Comparing seasons:**
- Use the same assist weight and turnover penalty across years for fair comparison
- Be aware that rule changes between seasons can affect stats
- Consider league expansion effects on player workloads

**Finding trends:**
- Look for players with consistent improvement across multiple years
- Identify role changes that correlate with performance gains
- Compare team role distributions to playoff success

**Exporting data:**
- Download CSV includes all calculated metrics for further analysis
- File names include the year for easy organization
