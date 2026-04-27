# Default Prediction Model — Data Specification
## Purpose
Build a firm-year panel with features and a binary default label.
- **Training data**: all firm-years up to and including 2025, with known outcomes
- **Validation/prediction data**: all active firms as of end-2025, predict default probability for 2026+
- **Output**: default probability per company, SHAP feature importance

---

## CRITICAL: No-Leak Rules
1. All financial features must use data from fiscal year T to predict default in year T+1 or later.
2. The default label for firm-year T is: did this firm default at any point between T+1 and T+3?
3. Never use post-event data (e.g., delisting price, delist reason) as a feature — only as the label.
4. Macro controls (Fed rates) are joined on the fiscal year-end date of the observation, not the default date.
5. Analyst signals (I/B/E/S) must use the `statpers` (statistics period) that falls before the fiscal year-end, not after.
6. The CCM link table must be filtered by date: only use a PERMNO-GVKEY link that was active during the observation year.

---

## TABLE 1: crsp_a_stock.wrds_msfv2_query
**Purpose**: Market data — price, return, market cap, volatility inputs. One row per PERMNO per month.

### Columns to pull:
| Column | Use |
|--------|-----|
| `permno` | Primary key — joins to CCM link table |
| `mthcaldt` | Date — used to aggregate to annual, and to compute firm age |
| `mthprc` | Monthly price — take abs(mthprc), used to compute market cap |
| `shrout` | Shares outstanding — market cap = abs(mthprc) * shrout |
| `mthcap` | Monthly market capitalization (pre-computed, use as cross-check) |
| `mthret` | Monthly total return — used to compute 12-month trailing return and volatility |
| `mthretx` | Monthly return ex-dividends — use for price-only return series |
| `mthdelflg` | Monthly delisting flag — signals final observation month |
| `siccd` | SIC code — industry control |
| `naics` | NAICS code — industry control (alternative) |
| `primaryexch` | Primary exchange (NYSE/NASDAQ/AMEX) — control variable |
| `ticker` | Ticker — for reference/debugging only, do not use as join key |
| `vwretd` | Value-weighted market return — used to compute excess return (mthret - vwretd) |

### Columns to EXCLUDE (not needed):
`mthprcflg`, `mthprcdt`, `mthdtflg`, `mthcompflg`, `mthcompsubflg`, `mthprevprc`, `mthprevprcdt`, `mthprevdtflg`, `mthprevcap`, `mthretflg`, `mthdiscnt`, `mthvol`, `mthvolflg`, `mthprcvol`, `mthfacshrflg`, `mthprcvolmisscnt`, `shrstartdt`, `shrenddt`, `shrsource`, `shrfactype`, `shradrflg`, all `dis*` distribution fields, `nasdcompno`, `nasdissuno`, `vwretx`, `ewretd`, `ewretx`, `permco`, `cusip`, `hdrcusip`, `securitynm`, `shareclass`, `usincflg`, `issuertype`, `securitytype`, `securitysubtype`, `sharetype`, `tradingsymbol`, `icbindustry`, `exchangetier`, `conditionaltype`, `tradingstatusflg`

### Aggregation to annual:
```
For each PERMNO × fiscal_year:
  - mktcap_dec      = abs(mthprc) * shrout at December month-end
  - ret_12m         = product of (1 + mthret) over prior 12 months - 1
  - ret_vol_12m     = std(mthret) over prior 12 months  [annualize * sqrt(12)]
  - excess_ret_12m  = ret_12m minus sum of vwretd over same window
  - Keep siccd, primaryexch from most recent month
```

---

## TABLE 2: crsp_a_stock.stkdelists
**Purpose**: Default label construction. One row per PERMNO delist event.
This table provides the outcome variable — it is NEVER used as a feature.

### Columns to pull:
| Column | Use |
|--------|-----|
| `permno` | Join key to MSF and CCM |
| `delistingdt` | Date of delisting — used to compute years_to_default |
| `delreasontype` | String reason for delisting — **primary default classifier** |
| `delactiontype` | Broad action category (Merger / Bankruptcy / Liquidation etc.) |
| `delstatustype` | Completion status — filter to completed delistings only |
| `delret` | Delisting return — used for Shumway correction, NOT a feature |
| `delretmisstype` | Why return is missing — guides correction method |
| `delpermno` | Acquiring firm PERMNO — confirms clean M&A vs distress |
| `delpermco` | Acquiring firm PERMCO |
| `delnextprc` | Next price after delist — used to compute delret when missing |

### Default label construction:
```
default_flag = 1 if delreasontype IN (
    'Bankruptcy', 'Liquidation', 'Insufficient Capital',
    'Distressed', 'Chapter 11', 'Chapter 7'
)
-- exact strings: run SELECT DISTINCT delreasontype first to confirm values

default_flag = 0 if delreasontype IN (
    'Merger', 'Acquisition', 'Going Private', 'Voluntary',
    'Exchange Transfer', 'Price Below Minimum'
)

-- For firm-year T: default_label_1yr = 1 if firm defaults within 12 months of fiscal year-end T
-- For firm-year T: default_label_3yr = 1 if firm defaults within 36 months of fiscal year-end T
```

### Columns to EXCLUDE:
`deldtprc`, `deldtprcflg`, `delpaymenttype`, `delamtdt`, `deldivamt`, `deldistype`, `deldlydt`, `delnextdt`, `delnextprcflg`, `nasdissuno`, `siccd`, `primaryexch`

---

## TABLE 3: comp_na_daily_all.funda
**Purpose**: All financial statement features — income statement, balance sheet, cash flow. One row per GVKEY per fiscal year.

### Columns to pull:

**Identifiers and metadata:**
| Column | Use |
|--------|-----|
| `gvkey` | Primary key — joins to CCM link table |
| `datadate` | Fiscal year end date — critical for no-leak alignment |
| `fyear` | Fiscal year integer |
| `fyr` | Fiscal year end month — needed for calendar alignment |
| `tic` | Ticker (reference only) |
| `cusip` | CUSIP (reference only) |
| `cik` | CIK — joins to FJC bankruptcy table |
| `sic` | SIC code — industry fixed effect |
| `naics` | NAICS code |
| `gsector` | GICS sector — cleaner than SIC for industry controls |
| `gind` | GICS industry |
| `exchg` | Exchange code |
| `fic` | Country of incorporation — filter to US (fic = 'USA') |
| `loc` | Country of headquarters |
| `ipodate` | IPO date — compute firm_age = datadate - ipodate |
| `dlrsn` | Compustat deletion reason — cross-check on default label |
| `dldte` | Compustat deletion date |
| `costat` | Company status (A=active, I=inactive) |

**Balance sheet — assets:**
| Column | Description |
|--------|-------------|
| `at` | Total assets |
| `act` | Current assets total |
| `che` | Cash and short-term investments |
| `rect` | Receivables total |
| `invt` | Inventories total |
| `ppent` | PP&E net |
| `gdwl` | Goodwill |
| `intan` | Intangible assets total |
| `ivao` | Investments and advances other |

**Balance sheet — liabilities:**
| Column | Description |
|--------|-------------|
| `lt` | Total liabilities |
| `lct` | Current liabilities total |
| `dlc` | Debt in current liabilities (short-term debt) |
| `dltt` | Long-term debt total |
| `ap` | Accounts payable |
| `txp` | Income taxes payable |

**Balance sheet — equity:**
| Column | Description |
|--------|-------------|
| `ceq` | Common equity total |
| `re` | Retained earnings |
| `seq` | Stockholders equity parent |
| `csho` | Common shares outstanding |
| `pstk` | Preferred stock total |

**Income statement:**
| Column | Description |
|--------|-------------|
| `revt` | Revenue total (use `sale` if `revt` missing) |
| `sale` | Sales/turnover |
| `cogs` | Cost of goods sold |
| `gp` | Gross profit |
| `ebitda` | EBITDA (reported) |
| `oibdp` | Operating income before depreciation (EBITDA alternative) |
| `oiadp` | Operating income after depreciation (EBIT) |
| `ebit` | EBIT |
| `xint` | Interest expense total |
| `pi` | Pretax income |
| `txt` | Income taxes total |
| `ni` | Net income |
| `dp` | Depreciation and amortization |
| `xsga` | SG&A expense |
| `xrd` | R&D expense |

**Cash flow statement:**
| Column | Description |
|--------|-------------|
| `oancf` | Operating cash flow |
| `capx` | Capital expenditures |
| `dltis` | Long-term debt issuance |
| `dltr` | Long-term debt reduction |
| `sstk` | Sale of common and preferred stock |
| `prstkc` | Purchase of common and preferred stock (buybacks) |
| `dv` | Cash dividends |

**Supplemental:**
| Column | Description |
|--------|-------------|
| `mkvalt` | Market value total fiscal (cross-check) |
| `prcc_f` | Price close fiscal year-end |
| `csho` | Shares outstanding |
| `au` | Auditor code |
| `auop` | Auditor opinion — **going concern flag** (value 2 or 3 = going concern) |
| `emp` | Employees |

### Features derived from FUNDA:
```python
# Solvency / leverage
debt_to_ebitda       = (dlc + dltt) / ebitda
interest_coverage    = ebitda / xint                    # Altman Z4
net_debt_to_assets   = (dlc + dltt - che) / at
current_ratio        = act / lct
quick_ratio          = (act - invt) / lct
debt_to_equity       = (dlc + dltt) / ceq
debt_to_assets       = lt / at

# Altman Z-score components (keep raw, model learns weights)
wc_to_assets         = (act - lct) / at                 # Z1
re_to_assets         = re / at                          # Z2
ebit_to_assets       = ebit / at                        # Z3
equity_to_liabilities = (csho * prcc_f) / lt            # Z4 (market version)
sales_to_assets      = sale / at                        # Z5

# Profitability
roa                  = ni / at
roe                  = ni / ceq
ebitda_margin        = ebitda / sale
net_margin           = ni / sale
gross_margin         = gp / sale

# Cash flow
fcf                  = oancf - capx
fcf_to_debt          = fcf / (dlc + dltt)
cfo_to_assets        = oancf / at

# Size / growth
log_assets           = log(at)
revenue_growth       = (sale_t - sale_t1) / abs(sale_t1)
firm_age             = datadate - ipodate (years)

# Quality flags
going_concern        = 1 if auop IN (2, 3) else 0
```

### Columns to EXCLUDE from FUNDA (not needed):
All insurance-specific fields (`iafici`, `prvt`, `rvbci` etc.), all utility-specific fields (`uaox`, `uceq` etc.), all banking-specific fields (`lcacl`, `rll` etc.), pension detail fields (`pvpl` etc.), all `opt*` stock option fields, all `dvp*` preferred dividend detail, `add1`-`add4`, `phone`, `fax`, `weburl`, `ein`, `busdesc`.

---

## TABLE 4: comp_na_daily_all.fundq
**Purpose**: Quarterly financial features for the 4 quarters preceding default. Captures deterioration that annual data misses.

### Columns to pull (same logic as FUNDA, quarterly versions):
| Column | Description |
|--------|-------------|
| `gvkey` | Join key |
| `datadate` | Quarter end date |
| `fyearq` | Fiscal year |
| `fqtr` | Fiscal quarter (1-4) |
| `rdq` | Report date — use this for no-leak alignment (not datadate) |
| `atq` | Total assets |
| `actq` | Current assets |
| `cheq` | Cash and short-term investments |
| `rectq` | Receivables |
| `invtq` | Inventories |
| `lctq` | Current liabilities |
| `dlcq` | Short-term debt |
| `dlttq` | Long-term debt |
| `ceqq` | Common equity |
| `req` | Retained earnings |
| `revtq` | Revenue |
| `saleq` | Sales |
| `cogsq` | COGS |
| `oibdpq` | EBITDA (operating income before D&A) |
| `oiadpq` | EBIT (operating income after D&A) |
| `xintq` | Interest expense |
| `niq` | Net income |
| `dpq` | D&A |
| `oancfy` | Operating cash flow (YTD) |
| `capxy` | Capex (YTD) |
| `cshoq` | Shares outstanding |

### Features derived from FUNDQ:
```python
# Quarter-over-quarter changes (strong distress signals)
cash_burn_rate     = (cheq_t - cheq_t4) / cheq_t4      # 4-quarter cash decline
revenue_decay      = (saleq_t - saleq_t4) / saleq_t4    # 4-quarter revenue change
ni_trend           = [niq_t, niq_t1, niq_t2, niq_t3]   # earnings trajectory
working_cap_change = (actq - lctq)_t - (actq - lctq)_t4 # working capital deterioration
debt_buildup       = (dlcq + dlttq)_t - (dlcq + dlttq)_t4
```

### No-leak rule for FUNDQ:
Use `rdq` (report date) not `datadate`. A firm's Q3 earnings released in November cannot be used to predict an event before that November date.

---

## TABLE 5: wrdsapps_finratio.firm_ratio
**Purpose**: Pre-computed financial ratios — use as convenience layer and cross-check. Do NOT use as sole source; always verify against your own FUNDA-derived ratios.

### Columns to pull:
| Column | Description |
|--------|-------------|
| `gvkey` | Join key |
| `public_date` | Date ratios are valid as of |
| `debt_ebitda` | Total debt / EBITDA |
| `intcov_ratio` | Interest coverage ratio |
| `curr_ratio` | Current ratio |
| `quick_ratio` | Quick ratio |
| `de_ratio` | Debt / equity |
| `debt_assets` | Total liabilities / total assets |
| `roa` | Return on assets |
| `roe` | Return on equity |
| `npm` | Net profit margin |
| `opmad` | Operating profit margin after depreciation |
| `fcf_ocf` | FCF / operating cash flow |
| `cash_debt` | Cash flow / total debt |
| `lt_debt` | Long-term debt / total liabilities |
| `cash_lt` | Cash / total liabilities |

### Columns to EXCLUDE:
`capei` (Shiller CAPE — not firm-level), `divyield`, `dpr`, `peg_trailing`, all valuation multiples (`evm`, `pcf`, `pe_op_basic`, `pe_op_dil`, `ps`, `ptb`, `bm`) — these use market prices which are already captured from CRSP. Including them here would double-count.

---

## TABLE 6: crsp_a_ccm.ccmxpf_lnkhist
**Purpose**: Bridge table — maps GVKEY (Compustat) to PERMNO (CRSP). Every CRSP-Compustat join goes through this table.

### Columns to pull:
| Column | Use |
|--------|-----|
| `gvkey` | Compustat identifier |
| `lpermno` | CRSP PERMNO — the join key to all CRSP tables |
| `lpermco` | CRSP PERMCO |
| `linktype` | Link quality type — **filter: keep only 'LC', 'LU', 'LS'** |
| `linkprim` | Primary link marker — **filter: keep only 'P', 'C'** |
| `linkdt` | First effective date of this link |
| `linkenddt` | Last effective date — treat NULL as '2099-12-31' |
| `cik` | CIK number — joins to FJC bankruptcy table |
| `gsector` | GICS sector |
| `gind` | GICS industry |
| `ggroup` | GICS group |
| `gsubind` | GICS sub-industry |
| `sic` | SIC code |
| `fyrc` | Fiscal year end month |
| `ipodate` | IPO date |
| `dlrsn` | Compustat deletion reason |
| `dldte` | Compustat deletion date |
| `costat` | Company status (A=active, I=inactive) |
| `fic` | Country of incorporation |
| `loc` | Country of headquarters |

### Filter logic (CRITICAL):
```python
ccm = ccm[
    (ccm['linktype'].isin(['LC', 'LU', 'LS'])) &
    (ccm['linkprim'].isin(['P', 'C']))
]
ccm['linkenddt'] = ccm['linkenddt'].fillna(pd.Timestamp('2099-12-31'))

# When joining to a firm-year observation:
# observation_date must be >= linkdt AND <= linkenddt
merged = merged[
    (merged['datadate'] >= merged['linkdt']) &
    (merged['datadate'] <= merged['linkenddt'])
]
```

### Columns to EXCLUDE:
`conm`, `conml`, `tic`, `cusip`, `liid`, `add1`-`add4`, `addzip`, `city`, `county`, `state`, `phone`, `fax`, `weburl`, `ein`, `busdesc`, `idbflag`, `incorp`, `stko`, `spcindcd`, `spcseccd`, `prican`, `prirow`, `priusa`, `naics`

---

## TABLE 7: fjc_litigation.bankruptcy
**Purpose**: Primary default label source (1929–2019). One row per bankruptcy case.
This table provides outcome labels only — never features.

### Columns to pull:
| Column | Use |
|--------|-----|
| `casekey` | Unique case identifier |
| `filedate` | Bankruptcy filing date — **true default event date** |
| `filecy` | Filing calendar year |
| `orgflchp` | Original filing chapter (7 or 11) — label: liquidation vs reorganization |
| `crntchp` | Current chapter — captures Chapter 11 → 7 conversions |
| `closedt` | Case closing date |
| `d1fdsp` | Debtor 1 final disposition (Discharged / Dismissed / Converted) |
| `d1fdspdt` | Final disposition date |
| `orgd1fprse` | Original debtor name prose — used for name matching if CIK missing |
| `d1fprse` | Current debtor name |
| `totassts` | Total assets at filing — sanity check for matching |
| `totlblts` | Total liabilities at filing |
| `district` | Court district |
| `prepack` | Prepackaged case flag — prepackaged Ch11 is less severe |
| `involuntary_flag` | Involuntary bankruptcy flag |
| `ticker` | Ticker if available — supplementary join key |
| `cusip` | CUSIP if available — supplementary join key |

### Join to Compustat/CRSP:
```
PRIMARY JOIN:   fjc.bankruptcy → ccm.cik  (via fjc_linking.wrds_bankruptcy_link)
FALLBACK JOIN:  fjc.bankruptcy.cik → comp_na_daily_all.funda.cik
                (direct CIK join, post-1996 only, ~85% coverage)
FINAL CHECK:    cross-validate against crsp.stkdelists on permno + date proximity
```

### Label construction:
```python
# default_type:
# 1 = Chapter 7 (liquidation) — more severe
# 2 = Chapter 11 (reorganization) — may emerge
# 3 = Chapter 11 converted to Chapter 7

default_flag = 1 for all rows in this table (it IS the bankruptcy event)
default_date = filedate  # use filing date, not closing date
```

### Columns to EXCLUDE:
`d1zip`, `d1cnty`, `d2zip`, `d2cnty`, `d2fprse`, `d2fdsp`, `d2fdspdt`, `d2fdspcy`, `d2fdspfy`, `d2chgdt`, `orgfeests`, `feests`, `orgdbtrtyp`, `dbtrtyp`, `smllbus`, `realprop`, `persprop`, `secured`, `unsecpr`, `unsecnpr`, `dschrgd`, `ndschrgd`, `totdbt`, `cntmnthi`, `avgmnthi`, `avgmnthe`, `srccase`, `dstncase`, `cnsllead`, `jntlead`, `flcmecfv`, `clcmecfv`, `taxexempt`, `sec304_flag`, `assetcase`, `c11dvdnd`, `c11ftrpay`, all `circuit`, `office`, `gen`, `seq`, `origin`, `docket` fields

---

## TABLE 8: ciq_keydev.wrds_keydev
**Purpose**: Two uses — (1) bankruptcy labels post-2019 to fill FJC gap; (2) going concern and auditor change flags as features.

### Columns to pull:
| Column | Use |
|--------|-----|
| `companyid` | CapIQ company ID — join key within CapIQ |
| `gvkey` | GVKEY — direct join to Compustat |
| `keydevid` | Unique event identifier |
| `keydeveventtypeid` | Event type code — **filter criteria** |
| `eventtype` | Event type string description |
| `announcedate` | Date of event announcement |
| `announceddateutc` | UTC announcement date |
| `headline` | Event headline — for manual review of edge cases |

### Filter by keydeveventtypeid:
```python
# For DEFAULT LABELS (post-2019 bankruptcy supplement):
BANKRUPTCY_EVENTS = [192, 26, 74]
# 192 = Chapter 11 filing
# 26  = Bankruptcy/Receivership
# 74  = Liquidation

# For FEATURES (qualitative flags, all years):
QUALITATIVE_FLAGS = [31, 83, 192, 26]
# 31  = Auditor change           → auditor_change_flag = 1
# 83  = Financial restatement    → restatement_flag = 1
# 192 = Going concern opinion    → going_concern_flag = 1
# 26  = Bankruptcy/Receivership  → (label only, not feature)
```

### No-leak rule for CapIQ features:
```python
# Going concern flag for firm-year T:
# Use announcedate WITHIN fiscal year T (before fiscal year-end datadate_T)
# Do NOT use announcedate after datadate_T as a feature for year T

going_concern_flag = 1 if any keydev event (type 192) 
                    has announcedate <= datadate_T 
                    AND announcedate > datadate_T-1
```

### Columns to EXCLUDE:
`situation`, `keydevtoobjectroletypeid`, `objectroletype`, `announcetime`, `announcedatetimezone`, `enterdate`, `entertime`, `entereddateutc`, `lastmodifieddate`, `lastmodifieddateutc`, `mostimportantdateutc`, `speffectivedate`, `sptodate`, `companyname`

---

## TABLE 9: fjc_linking.wrds_bankruptcy_link
**Purpose**: Pre-built link table mapping FJC case records to Compustat GVKEYs and CIKs. Use this as the primary FJC join method for 1929–2019.

### Columns to pull:
| Column | Use |
|--------|-----|
| `casekey` | FJC case key — joins to fjc_litigation.bankruptcy |
| `gvkey` | Compustat GVKEY — joins to funda and ccm |
| `cik` | CIK — cross-check |
| `cusip` | CUSIP — cross-check |
| `ticker` | Ticker — cross-check |

### Join logic:
```python
# Step 1: join fjc.bankruptcy to fjc_linking on casekey
# Step 2: use gvkey from link table to join to compustat panel
# Step 3: for cases not in link table (post-2019 or unmatched):
#         fall back to direct CIK join via funda.cik or ccm.cik
```

---

## TABLE 10: frb_all.rates_monthly
**Purpose**: Macro controls for default rate analysis and model features.

### Columns to pull:
| Column | Use |
|--------|-----|
| `date` | Month-end date — join key |
| `aaa` | Moody's Aaa corporate yield |
| `baa` | Moody's Baa corporate yield |
| `fedfunds` | Federal funds rate |
| `gs10` | 10-year Treasury constant maturity |
| `tb3ms` | 3-month T-bill secondary market |

### Features derived:
```python
credit_spread = baa - aaa          # Primary credit cycle control
term_spread   = gs10 - tb3ms       # Yield curve shape (recession predictor)
# fedfunds    = level of rates (monetary policy tightness)
```

### Aggregation and join:
```python
# Aggregate to annual: take December value or full-year mean
fed_annual = fed.groupby(fed['date'].dt.year)[
    ['aaa','baa','fedfunds','gs10','tb3ms']
].mean()
fed_annual['credit_spread'] = fed_annual['baa'] - fed_annual['aaa']
fed_annual['term_spread']   = fed_annual['gs10'] - fed_annual['tb3ms']

# Join to firm-year panel on fyear (fiscal year integer)
panel = panel.merge(fed_annual, left_on='fyear', right_on='year')
```

### All other columns: EXCLUDE
Everything except the 5 columns listed above.

---

## TABLE 11: tr_ibes.statsum_epsus
**Purpose**: Analyst consensus signals — EPS estimate revisions and coverage changes as forward-looking distress indicators.

### Columns to pull:
| Column | Use |
|--------|-----|
| `ticker` | I/B/E/S ticker — join key (use oftic for official ticker) |
| `oftic` | Official ticker — preferred join key |
| `cusip` | CUSIP (8-digit) — secondary join key |
| `statpers` | Statistics period date — **use this as the observation date** |
| `fpedats` | Forecast period end date — identifies which fiscal year is being forecast |
| `fiscalp` | Periodicity (ANN/QTR) |
| `measure` | Measure type (EPS / SAL / NET) |
| `fpi` | Forecast period indicator (1=FY1, 6=Q1) |
| `numest` | Number of analysts — use to compute coverage change |
| `meanest` | Consensus mean estimate — use to compute revision |
| `medest` | Median estimate — more robust than mean |
| `stdev` | Standard deviation — analyst disagreement / uncertainty signal |
| `numup` | Number of upward revisions |
| `numdown` | Number of downward revisions |
| `actual` | Realized value — used to compute earnings surprise |
| `anndats_act` | Actual announcement date |
| `usfirm` | US firm flag — filter to 1 |

### Filter settings:
```
measure IN ('EPS', 'SAL', 'NET')
fpi IN (1, 6)        -- FY1 annual and Q1 quarterly
usfirm = 1           -- US firms only
fiscalp = 'ANN'      -- for fpi=1; 'QTR' for fpi=6
```

### Features derived:
```python
# Join I/B/E/S to firm-year panel:
# Match statpers to the 3-month window BEFORE fiscal year-end datadate
# i.e., statpers BETWEEN (datadate - 90 days) AND datadate

# EPS estimate revision (QoQ)
eps_revision     = meanest_t - meanest_t1    # change vs prior period
eps_revision_pct = eps_revision / abs(meanest_t1)

# Analyst coverage change
coverage_change  = numest_t - numest_t4      # 4-period change
coverage_drop    = 1 if numest_t < numest_t4 * 0.75  # 25%+ analyst drop

# Estimate dispersion
est_dispersion   = stdev / abs(meanest)      # coefficient of variation

# Revision direction
net_revision_dir = (numup - numdown) / numest   # net % of upward vs downward revisions

# Earnings surprise (realized vs expected)
earnings_surprise = (actual - meanest) / abs(meanest)
```

### No-leak rule for I/B/E/S:
```python
# For firm-year T prediction:
# Use statpers that falls BEFORE datadate_T (fiscal year-end)
# The most recent statpers before datadate_T is the valid observation

ibes_for_year_T = ibes[
    (ibes['fpedats'] == datadate_T) &    # forecast is for this fiscal year
    (ibes['statpers'] <= datadate_T)     # consensus known before year-end
].sort_values('statpers').groupby(['cusip','fpi','measure']).last()
```

### Columns to EXCLUDE:
`cname`, `highest`, `lowest`, `curcode`, `anntims_act`, `curr_act`

---

## FULL JOIN CHAIN

```
MASTER PANEL CONSTRUCTION (one row = one GVKEY × fiscal year):

Step 1 — BACKBONE
  comp_na_daily_all.funda
  ON gvkey, datadate
  → one row per company per fiscal year, financials only

Step 2 — ADD CRSP MARKET DATA
  funda
  JOIN crsp_a_ccm.ccmxpf_lnkhist
      ON funda.gvkey = ccm.gvkey
      AND funda.datadate BETWEEN ccm.linkdt AND ccm.linkenddt
      AND ccm.linktype IN ('LC','LU','LS')
      AND ccm.linkprim IN ('P','C')
  → adds lpermno (CRSP PERMNO)

  JOIN crsp_a_stock.wrds_msfv2_query (aggregated to annual)
      ON crsp.permno = ccm.lpermno
      AND crsp.fiscal_year = funda.fyear
  → adds mktcap, ret_12m, ret_vol_12m, excess_ret_12m

Step 3 — ADD DEFAULT LABEL
  LEFT JOIN crsp_a_stock.stkdelists
      ON stkdelists.permno = ccm.lpermno
  → for each firm-year T:
      default_label_1yr = 1 if stkdelists.delistingdt BETWEEN datadate_T AND datadate_T + 365
                          AND delreasontype = bankruptcy/liquidation

  LEFT JOIN fjc_linking.wrds_bankruptcy_link → fjc_litigation.bankruptcy
      ON wrds_link.gvkey = funda.gvkey (for 1929-2019)
      OR fjc.cik = funda.cik (fallback for post-2019 / unmatched)
  → supplements/overrides label with true filing date

  LEFT JOIN ciq_keydev.wrds_keydev
      ON keydev.gvkey = funda.gvkey
      AND keydeveventtypeid IN (192, 26, 74)
      AND announcedate > datadate_T  (post-2019 gap fill)
  → fills gap for 2020+ defaults

  FINAL LABEL PRIORITY: FJC filing date > CapIQ event date > CRSP delist date

Step 4 — ADD QUARTERLY FEATURES
  LEFT JOIN comp_na_daily_all.fundq (last 4 quarters before datadate)
      ON fundq.gvkey = funda.gvkey
      AND fundq.rdq <= funda.datadate          ← no-leak: use report date
      AND fundq.rdq > funda.datadate - 365
  → adds quarterly trend features

Step 5 — ADD QUALITATIVE FLAGS
  LEFT JOIN ciq_keydev.wrds_keydev
      ON keydev.gvkey = funda.gvkey
      AND keydeveventtypeid IN (31, 83, 192)
      AND announcedate <= funda.datadate        ← no-leak: only prior to year-end
      AND announcedate > funda.datadate - 365
  → adds going_concern_flag, auditor_change_flag, restatement_flag

Step 6 — ADD MACRO CONTROLS
  JOIN frb_all.rates_monthly (aggregated to annual)
      ON frb.year = funda.fyear
  → adds credit_spread, term_spread, fedfunds

Step 7 — ADD ANALYST SIGNALS
  LEFT JOIN tr_ibes.statsum_epsus
      ON ibes.cusip = funda.cusip (8-digit match)
      AND ibes.fpedats = funda.datadate         ← forecast is for this year
      AND ibes.statpers <= funda.datadate        ← no-leak: known before year-end
      AND ibes.usfirm = 1
      AND ibes.fpi IN (1, 6)
      AND ibes.measure IN ('EPS','SAL','NET')
  → adds eps_revision, coverage_change, est_dispersion

Step 8 — OPTIONAL FINANCIAL RATIOS CROSS-CHECK
  LEFT JOIN wrdsapps_finratio.firm_ratio
      ON firm_ratio.gvkey = funda.gvkey
      AND firm_ratio.public_date <= funda.datadate   ← no-leak
  → cross-check ratios against own calculations
```

---

## FILTERS TO APPLY TO FINAL PANEL

```python
# 1. US-listed common equity only
panel = panel[panel['shrcd'].isin([10, 11])]          # CRSP share codes
panel = panel[panel['fic'] == 'USA']                   # US incorporated
panel = panel[panel['primaryexch'].isin(['N','A','Q'])] # NYSE, AMEX, NASDAQ

# 2. Exclude financial firms and utilities (different capital structure)
panel = panel[~panel['sic'].between(6000, 6999)]       # financials
panel = panel[~panel['sic'].between(4900, 4999)]       # utilities

# 3. Require minimum size (avoid penny stocks / shell companies)
panel = panel[panel['mktcap_dec'] >= 10]               # $10M market cap minimum
panel = panel[panel['at'] >= 1]                        # $1M assets minimum

# 4. Require at least 2 years of history
panel = panel[panel['firm_age'] >= 2]

# 5. Training vs prediction split
train = panel[panel['fyear'] <= 2024]                  # known outcomes
predict = panel[panel['fyear'] == 2025]                # 2026 prediction targets
```

---

## FINAL FEATURE LIST (model inputs)

### From CRSP (market):
- `mktcap_dec` — market capitalization (log-transformed)
- `ret_12m` — 12-month trailing return
- `ret_vol_12m` — 12-month return volatility
- `excess_ret_12m` — return minus market return

### From FUNDA (financial):
**Leverage:** `debt_to_ebitda`, `interest_coverage`, `net_debt_to_assets`, `debt_to_equity`, `debt_to_assets`
**Liquidity:** `current_ratio`, `quick_ratio`, `cash_to_assets` (che/at)
**Altman Z:** `wc_to_assets`, `re_to_assets`, `ebit_to_assets`, `equity_to_liabilities`, `sales_to_assets`
**Profitability:** `roa`, `ebitda_margin`, `net_margin`, `gross_margin`
**Cash flow:** `fcf_to_debt`, `cfo_to_assets`
**Size/structure:** `log_assets`, `firm_age`, `revenue_growth`

### From FUNDQ (quarterly trends):
- `cash_burn_rate` — 4-quarter cash decline
- `revenue_decay` — 4-quarter revenue change
- `working_cap_change` — working capital deterioration
- `debt_buildup` — 4-quarter debt increase

### From CapIQ (qualitative):
- `going_concern_flag` — auditor going concern opinion
- `auditor_change_flag` — auditor switched during year
- `restatement_flag` — financial restatement during year

### From Fed H.15 (macro):
- `credit_spread` — Baa minus Aaa yield
- `term_spread` — 10yr minus 3m Treasury
- `fedfunds` — fed funds rate level

### From I/B/E/S (analyst):
- `eps_revision_pct` — consensus EPS revision
- `coverage_change` — analyst count change
- `est_dispersion` — estimate standard deviation / mean
- `net_revision_dir` — net upgrades minus downgrades ratio

### Target variable:
- `default_label_1yr` — defaulted within 12 months (primary)
- `default_label_3yr` — defaulted within 36 months (secondary)
- `default_type` — 1=Chapter7, 2=Chapter11, 0=no default (for stratified analysis)
