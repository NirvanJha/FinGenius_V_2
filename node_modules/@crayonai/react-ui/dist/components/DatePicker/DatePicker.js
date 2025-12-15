import { jsx as _jsx } from "react/jsx-runtime";
import { useState } from "react";
import { useLayoutContext } from "../../context/LayoutContext";
import { FloatingDatePickerRenderer } from "./helpers/components/FloatingDatePickerRenderer";
import { DatePickerProvider } from "./helpers/context/DatePickerContext";
const DatePicker = (props) => {
    const { layout } = useLayoutContext();
    const { mode = "single", selectedSingleDate, selectedRangeDates, setSelectedSingleDate, setSelectedRangeDates, isOpen, setIsOpen, className, style, } = props;
    const [internalSelectedDate, setInternalSelectedDate] = useState(selectedSingleDate);
    const [internalSelectedRange, setInternalSelectedRange] = useState(selectedRangeDates);
    const [internalIsOpen, setInternalIsOpen] = useState(isOpen ?? false);
    // this derived setter state is used to make this component more flexible
    // it allows the user to pass in a setter function from the parent component
    // if the parent component does not pass in a setter function, the internal setter state is used
    // this allows the user to pass in a setter function from the parent component, but still have the internal state be set
    // this design decision closely follow
    const selectedDateHandler = (selectedDate) => {
        if (selectedSingleDate) {
            setSelectedSingleDate?.(selectedDate);
            return;
        }
        setSelectedSingleDate?.(selectedDate);
        setInternalSelectedDate(selectedDate);
    };
    const selectedRangeHandler = (selectedRange) => {
        if (selectedRangeDates) {
            setSelectedRangeDates?.(selectedRange);
            return;
        }
        setSelectedRangeDates?.(selectedRange);
        setInternalSelectedRange(selectedRange);
    };
    return (_jsx(DatePickerProvider, { mode: mode, botType: layout, selectedDateFromParent: selectedSingleDate ?? internalSelectedDate, selectedRangeFromParent: selectedRangeDates ?? internalSelectedRange, setSelectedDateFromParent: selectedDateHandler, setSelectedRangeFromParent: selectedRangeHandler, isOpenFromParent: isOpen ?? internalIsOpen, setIsOpenFromParent: setIsOpen ?? setInternalIsOpen, children: _jsx(FloatingDatePickerRenderer, { className: className, style: style }) }));
};
export { DatePicker };
//# sourceMappingURL=DatePicker.js.map