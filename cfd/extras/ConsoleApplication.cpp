// ConsoleApplication1.cpp : This file contains the 'main' function. Program execution begins and
// ends there.
//

#include <iostream>
#include <list>
#include <type_traits>
#include <utility>
#include <vector>

#include "IterTools.h"

template<typename C>
void print_container(const C& container)
{
  typename C::const_iterator it = container.begin();

  while (it != container.end()) {
    const typename C::value_type& element = *it;
    std::cout << element << "\n";
    ++it;
  }

  std::cout << "done\n";
}

int main()
{
  SquareRange<int> xbounds {-1, 5};
  SquareRange<int> ybounds {0, 4};
  SquareRange<int> zbounds {1, 3};

  // In this callback, "min change level" refers to the highest iteration level
  // that changed in the last state transition.  It's 'min' because more significant levels
  // are earlier in the array/stack.  For example, if minChangeLevel == 0 in the function below
  // then there was a change in the xbound.

  // In real applications, you can use the min change level to recalculate loop state that remains
  // constant for less significant levels.  For example, if you have some value depending only on x
  // and y, but not z, you can recalculate it whenever minChangeLevel <= 1 and leave it be in other
  // cases

  auto printIter = [](const std::array<IterState<int>, 3>& iterState, size_t minChangeLevel)
  {
    std::cout << "iter state: ";
    for (size_t i = 0; i < iterState.size(); ++i) {
      std::cout << iterState[i].cur << ", ";
    }
    std::cout << "; min change level = " << minChangeLevel << '\n';
    return true;
  };

  NestedIterationGen<int, 3>(printIter, xbounds, ybounds, zbounds);

  std::cout << "\n\n\n";

  // Now create a simple iteration of all 3-element subsets of a set of 6 elements
  // There will be C(6,3) = 20 outputs, as expected
  SquareRange<int> setRange {1, 7};
  TriangleRange<int> firstBound {1};
  TriangleRange<int> secondBound {1};

  NestedIteration<int>(printIter, setRange, firstBound, secondBound);

  std::string s;

  // const std::string&

  s = std::string("blah");
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started:
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add
//   existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln
//   file
