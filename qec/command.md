Example for zz parity check data:
```bash
python qec/generate_dataset.py --phase all --source processed_zz_parity_check/qec_table.csv --target-columns meas_D1 meas_D2 log_flip --hierarchical-columns shot_id round_id prep_state prep_D1 prep_D2 expected_parity parity --output generated/qec.csv --model-path saved_models/QEC/model_ordinal.pth
```
