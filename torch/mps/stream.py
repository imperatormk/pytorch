import torch


class Stream(torch._C._MPSStreamBase):
    r"""Wrapper around an MPS stream.

    An MPS stream is a linear sequence of execution that belongs to a specific
    device, independent of other streams. Each stream owns its own Metal command
    queue, so work submitted to different streams may run concurrently.

    Streams come from a fixed pool that lives for the lifetime of the process;
    constructing a :class:`Stream` hands out the next one in round-robin order.

    Args:
        device (torch.device or int, optional): ignored, present for parity with
            the other backends. MPS is single-device.
    """

    def __new__(cls, device=None, **kwargs):
        return super().__new__(cls)

    # An MPSEvent binds its stream when it is acquired, not when it is
    # recorded -- rebinding a live event would reset its signal counter
    # (MPSEvent::reset). So an event created under this stream already records
    # here, while an event handed in from elsewhere has to be waited on with
    # this stream selected.
    def wait_event(self, event) -> None:
        r"""Make all future work submitted to this stream wait for an event.

        Args:
            event (torch.mps.Event): an event to wait for.
        """
        with stream(self):
            event.wait()

    def wait_stream(self, other) -> None:
        r"""Synchronize with another stream.

        All future work submitted to this stream will wait until all kernels
        already submitted to the given stream have completed.

        Args:
            other (Stream): a stream to synchronize with.
        """
        self.wait_event(other.record_event())

    def record_event(self, event=None):
        r"""Record an event on this stream.

        Args:
            event (torch.mps.Event, optional): event to record. If not given, a
                new one is allocated on this stream.

        Returns:
            Recorded event.
        """
        from .event import Event

        with stream(self):
            if event is None:
                event = Event()
            event.record()
        return event

    def synchronize(self) -> None:
        r"""Wait for all the kernels in this stream to complete."""
        super().synchronize()

    def __eq__(self, o) -> bool:
        if isinstance(o, Stream):
            return super().__eq__(o)
        return False

    def __hash__(self) -> int:
        return hash((self.stream_id, self.device_index, self.device_type))

    def __repr__(self) -> str:
        return f"<torch.mps.Stream device={self.device} stream_id={self.stream_id}>"


def current_stream() -> Stream:
    r"""Return the currently selected :class:`Stream`."""
    return torch.accelerator.current_stream()  # type: ignore[return-value]


def set_stream(stream: Stream) -> None:
    r"""Set the current stream. This is a wrapper API to set the stream.

    Usage of this function is discouraged in favor of the :func:`stream` context
    manager.

    Args:
        stream (Stream): selected stream. This function is a no-op if this
            argument is ``None``.
    """
    if stream is None:
        return
    torch.accelerator.set_stream(stream)


class StreamContext:
    r"""Context-manager that selects a given stream.

    All MPS kernels queued within its context will be enqueued on a selected
    stream.

    Args:
        stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    """

    cur_stream: "torch.mps.Stream | None"

    def __init__(self, stream: "torch.mps.Stream | None") -> None:
        self.stream = stream
        self.prev_stream = None

    def __enter__(self) -> None:
        if self.stream is None:
            return
        self.prev_stream = current_stream()
        set_stream(self.stream)

    def __exit__(self, type, value, traceback) -> None:
        if self.stream is None:
            return
        set_stream(self.prev_stream)


def stream(stream: "torch.mps.Stream | None") -> StreamContext:
    r"""Wrap the selection of an MPS stream in a context manager.

    Args:
        stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    """
    return StreamContext(stream)
