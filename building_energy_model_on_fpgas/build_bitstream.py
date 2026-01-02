# hls4ml conversion and FPGA build
import os
import numpy as np
import hls4ml
from sklearn.metrics import accuracy_score

from building_energy_model_on_fpgas.prepare_dataset import prepare_energy_dataset
from building_energy_model_on_fpgas.train_model import build_anomaly_model


def build_fpga_bitstream(csv_path):
    """
    Train QKeras anomaly detection model and
    convert it to FPGA bitstream using hls4ml.
    """

    # Prepare dataset
    X_train, X_test, y_train, y_test = prepare_energy_dataset(csv_path)

    # Build and train model
    model = build_anomaly_model(input_dim=X_train.shape[1])

    model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=30,
        validation_split=0.2,
        shuffle=True,
        verbose=1,
    )

    # CPU inference (sanity check)
    y_pred_cpu = (model.predict(X_test) > 0.5).astype(int)
    acc_cpu = accuracy_score(y_test, y_pred_cpu)
    print(f"CPU model accuracy: {acc_cpu * 100:.2f}%")

    # Create hls4ml config
    config = hls4ml.utils.config_from_keras_model(
        model, granularity="name"
    )

    # Output layer precision (important for sigmoid)
    config["LayerName"]["output"]["Precision"] = "ap_fixed<16,6>"
    config["LayerName"]["sigmoid"]["Precision"] = "ap_fixed<16,6>"

    # Reuse factor (controls area vs latency)
    for layer in ["fc1", "fc2", "output"]:
        config["LayerName"][layer]["ReuseFactor"] = 64

    # Convert to HLS model
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir="hls4ml_energy_anomaly_prj",
        backend="VivadoAccelerator",
        board="pynq-z2",
    )

    # Compile HLS model
    hls_model.compile()

    # HLS inference (CPU)
    y_pred_hls = (hls_model.predict(X_test) > 0.5).astype(int)
    acc_hls = accuracy_score(y_test, y_pred_hls)
    print(f"HLS model accuracy: {acc_hls * 100:.2f}%")

    # Build bitstream
    hls_model.build(
        csim=False,
        synth=True,
        export=True,
        bitfile=True,
    )

    print("✅ FPGA bitstream generation completed.")


if __name__ == "__main__":
    # Update this path to your dataset CSV
    build_fpga_bitstream("data/building_energy.csv")
