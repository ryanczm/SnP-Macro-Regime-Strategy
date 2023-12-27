## S&P Macro Regime Strategy

* Project Structure
    * Backtest and strategy: `economic_regime_avm.ipynb`
    * Helper files: `macro_utils.py`
    * Data: `data.xlsx`
* Description
    * A project ([post here](https://ryan-chew.com/sp_macro_strategy.html)) to come up with a strategy to beat long S&P based on rebalancing SPX, Gold, and US10Y based on macro conditions. 
    * Went with quarterly rebalancing over based on regimes of rising/falling growth/inflation from GDP/CPI data.
    * Classified in-sample period into regime quadrants, calculate Sharpes of each instrument for each regime to use as an alpha for portfolio weights.

<br>
<br>
<p align='center'>
<img src="img_sp.png" height="500">
<p align='center'>S&P vs Gold</p align='center'>
</p>
<br>
<p align='center'>
<img src="img_macro.png" height="500px">
<p align='center'>Year on year changes in GDP/CPI.</p align='center'>
</p>
<br>
<p align='center'>
<img src="img_oos_rets.png" height="500px">
<p align='center'>Out-sample returns of rebalancing strategy.</p align='center'>
</p>
<br>
<p align='center'>
<img src="img_weights.png" height="500px">
<p align='center'>Signals for portfolio weights in each regime.</p align='center'>
</p>
