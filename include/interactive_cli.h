/**
 * @file interactive_cli.h
 * @brief Header file for Interactive Command Line Interface
 * 
 * This header defines the interface for the interactive CLI functionality
 * that provides a user-friendly menu-driven experience for training and
 * generation tasks.
 */

#ifndef INTERACTIVE_CLI_H
#define INTERACTIVE_CLI_H

#ifdef __cplusplus
extern "C" {
#endif

// Platform-specific executable name
#ifdef _WIN32
    #define CLI_EXECUTABLE_NAME "c_transformer.exe"
#else
    #define CLI_EXECUTABLE_NAME "./c_transformer"
#endif

/**
 * Main function to start the interactive CLI
 * This provides a menu-driven interface for users to:
 * - Train models with optimized settings
 * - Generate text using trained models
 * - View model status and configuration
 * - Access help and documentation
 */
void run_interactive_cli(void);

#ifdef __cplusplus
}
#endif

#endif /* INTERACTIVE_CLI_H */
