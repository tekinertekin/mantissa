#ifndef MANTISSA_EXPORT_H
#define MANTISSA_EXPORT_H

/* Cross-platform symbol visibility so the same sources build a clean .dll on
 * Windows and .so/.dylib on Unix for the Python (ctypes) binding. */
#if defined(_WIN32) || defined(__CYGWIN__)
  #ifdef TK_BUILD_DLL
    #define TK_API __declspec(dllexport)
  #else
    #define TK_API __declspec(dllimport)
  #endif
#else
  #define TK_API __attribute__((visibility("default")))
#endif

#endif /* MANTISSA_EXPORT_H */
