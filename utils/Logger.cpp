//
// Created by nico on 04.05.23.
//

#include "Logger.h"
#include <sstream>

const char* const Logger::logFile = "log.txt";
const char* const Logger::tab = "  "; // 2 spaces
std::string Logger::prefix = "";

std::string Logger::getCurrentDateTime() {
  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);
  std::ostringstream oss;
  oss << std::put_time(&tm, "%d-%m-%Y %H:%M:%S");
  return oss.str();
}

void Logger::log(const std::string& message, const char function[]) {
  const std::string output = prefix + tab + function + tab + message; // build message
  std::cout << output << std::endl; // write to console
  std::ofstream out; // write to log file
  try {
    out.open(logFile, std::ios_base::app);
    out << getCurrentDateTime() << tab << output << std::endl; // we only need time in log file
    out.close();
  } catch (const std::ofstream::failure& e) {
    std::cout << "Error: Could not open/write to log file" << std::endl;
  }
}

void Logger::setPrefix(const std::string &s) {
  Logger::prefix = s;
}
