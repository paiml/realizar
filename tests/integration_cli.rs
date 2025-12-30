//! Integration tests for CLI binary
//!
//! These tests verify the `realizar` CLI commands work correctly.

#![allow(deprecated)]

use std::process::Command;

use assert_cmd::{assert::OutputAssertExt, cargo::CommandCargoExt};
use predicates::prelude::*;

#[test]
fn test_cli_help() {
    let mut cmd = Command::cargo_bin("realizar").unwrap();
    cmd.arg("--help");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Usage: realizar"));
}

#[test]
fn test_cli_info_command() {
    let mut cmd = Command::cargo_bin("realizar").unwrap();
    cmd.arg("info");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Realizar"))
        .stdout(predicate::str::contains("v0.3.2"));
}

#[test]
fn test_cli_serve_requires_demo_or_model() {
    let mut cmd = Command::cargo_bin("realizar").unwrap();
    cmd.arg("serve");
    // Without --demo flag, it should fail since no model path is provided
    cmd.assert().failure();
}

#[test]
fn test_cli_serve_help() {
    let mut cmd = Command::cargo_bin("realizar").unwrap();
    cmd.arg("serve").arg("--help");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("--demo"))
        .stdout(predicate::str::contains("--port"))
        .stdout(predicate::str::contains("--host"));
}

#[test]
fn test_cli_serve_invalid_port() {
    let mut cmd = Command::cargo_bin("realizar").unwrap();
    cmd.arg("serve").arg("--demo").arg("--port").arg("invalid");
    cmd.assert().failure();
}

#[test]
fn test_cli_unknown_command() {
    let mut cmd = Command::cargo_bin("realizar").unwrap();
    cmd.arg("unknown");
    cmd.assert().failure();
}

#[test]
fn test_cli_version_flag() {
    let mut cmd = Command::cargo_bin("realizar").unwrap();
    cmd.arg("--version");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("realizar"));
}
