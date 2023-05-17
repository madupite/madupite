//
// Created by nico on 04.05.23.
//

#ifndef BATTLESHIP_LOGGER_H
#define BATTLESHIP_LOGGER_H

#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>
#include <ctime>

// Macro to include function name in log message
#define LOG(message) Logger::log(message, __func__)

/*!
 * Basic logging class. Only static usage
 */
class Logger {

public:

  enum class Type {
    Info,
    Warning,
    Error
  };

  /*!
   * outputs a message to std::cout and to the logfile
   * @param message text to be logged
   * @param type optional log type (info, warning, error). currently unused
   */
  static void log(const std::string &message, const char function[] = "");

  static void setPrefix(const std::string &s);

private:
  /*!
   * Helper function for logging
   * @return returns a formatted string timestamp
   */
  static std::string getCurrentDateTime();

  static std::string prefix;  ///< prefix for logging messages. intended to be "client" or "server"
  static const char* const logFile; ///< file path for the log file
  static const char* const tab;   ///< defines a tabulator (4 spaces)
};

#endif // BATTLESHIP_LOGGER_H
